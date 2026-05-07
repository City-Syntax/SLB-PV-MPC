# Author: Kaifeng ZHU.
# This version calculates the theoretical limit of the battery energy storage system.
import pandas as pd
from pack_slb import PackModelSeriesParallel
from battery_system_slb import BatterySystemSeriesParallel
import numpy as np

from joblib import Parallel, delayed

from cell_slb import create_cell_from_stored_json
from numba import njit, prange



def build_system(cache_dir: str,
                 Ns: int, Np: int, N_series: int, N_parallel: int):
    '''Build the battery system.'''
    cell = create_cell_from_stored_json(cache_dir, show_info=False)
    pack = PackModelSeriesParallel(cell=cell, Ns=Ns, Np=Np)
    system = BatterySystemSeriesParallel(pack=pack, N_series=N_series, N_parallel=N_parallel)
    return system

def get_fixed_soh(cell):
    """
    固定 SOH：用当前 efc_cycle 对 degradation_df 插值得到一个常数。
    之后整个 Step C 都用这个 soh，不更新 efc_cycle。
    """
    return float(np.interp(cell.efc_cycle, cell.degradation_df.index, cell.degradation_df))

def build_global_actions(system, soc_grid, action_grid_size=81):
    """
    构造全局动作网格：[-Pdis_global_max, +Pch_global_max]
    取 soc_grid 上的最大 envelope 作为全局边界。
    """
    pch_max_list = [system.get_step_power_limit_kw("charge", soc=float(s)) for s in soc_grid]
    pdis_max_list = [system.get_step_power_limit_kw("discharge", soc=float(s)) for s in soc_grid]
    PCH = float(np.max(pch_max_list))
    PDIS = float(np.max(pdis_max_list))
    actions = np.linspace(-PDIS, PCH, action_grid_size).astype(np.float32)
    return actions


def build_soc_dependent_actions(system, soc_grid, action_grid_size=81, strict_boundary=True):
    """
    构造与 SOC 相关的动作网格：在每个 soc 下只允许该 SOC 下物理可行的方向。
    - 在 soc_max：只允许放电和零（充电上限=0），避免满电时 DP 仍选“充电”。
    - 在 soc_min：只允许充电和零（放电上限=0）。
    - 中间 SOC：[-p_dis(soc), ..., +p_ch(soc)]。
    strict_boundary: 若 True，在边界 SOC 将反方向 limit 置 0；若 False，仍用 LUT 给出的小值。
    """
    cell = system.pack.cell
    soc_min = float(cell.soc_min)
    soc_max = float(cell.soc_max)
    tol = 1e-6

    Nsoc = len(soc_grid)
    actions_2d = np.empty((Nsoc, action_grid_size), dtype=np.float32)

    for j, s in enumerate(soc_grid):
        soc = float(s)
        p_ch = float(system.get_step_power_limit_kw("charge", soc=soc))
        p_dis = float(system.get_step_power_limit_kw("discharge", soc=soc))
        if strict_boundary:
            if soc >= soc_max - tol:
                p_ch = 0.0
            if soc <= soc_min + tol:
                p_dis = 0.0
        actions_2d[j, :] = np.linspace(-p_dis, p_ch, action_grid_size).astype(np.float32)

    return actions_2d

def transition_soc_ah_fixed_soh(system, soc0, p_request_sys_kw, soh_fixed, dt_s=3600.0):
    """
    纯状态转移：复刻 CellSLB.step() 的 SOC 更新逻辑（Ah/电流），但不修改 cell/system 内部状态。
    输入：
      - soc0: 当前 SOC
      - p_request_sys_kw: 系统功率请求(kW)，charge为正，discharge为负（与你模型一致）
      - soh_fixed: 固定 SOH
    输出：
      - soc_next
      - p_applied_sys_kw: 实际可施加功率(kW)，考虑 envelope + SOC/电压限制
    """
    cell = system.pack.cell
    dt_s = float(dt_s)

    # system(kW) -> pack(W) -> cell(W)
    p_req_sys_w = float(p_request_sys_kw) * 1000.0
    p_req_pack_w = p_req_sys_w / system.n_packs
    n_cells = system.pack.Ns * system.pack.Np
    p_req_cell_w = p_req_pack_w / n_cells

    # 1) envelope 限制（固定 soh_fixed + soc0）
    if (soc0 < cell.soc_min) or (soc0 > cell.soc_max):
        p_env_applied = 0.0
    else:
        if p_req_cell_w >= 0:
            p_env_lim = cell.p_max_charge_interp(soh_fixed, soc0)  # W
            p_env_applied = float(np.clip(p_req_cell_w, 0.0, p_env_lim))
        else:
            p_env_lim = cell.p_max_discharge_interp(soh_fixed, soc0)  # W magnitude
            p_env_applied = -float(np.clip(-p_req_cell_w, 0.0, p_env_lim))

    # 2) LUT 参数
    Voc0 = cell.luts["Voc"](soh_fixed, soc0)
    Rint0 = cell.luts["Rint"](soh_fixed, soc0)
    R10 = cell.luts["R1"](soh_fixed, soc0)
    R20 = cell.luts["R2"](soh_fixed, soc0)
    R_overall = Rint0 + R10 + R20

    Qa_Ah = max(cell.nominal_capacity_Ah * soh_fixed, 1e-9)

    # 3) SOC/电压限制导致的本步最大可施加功率（与 CellSLB.step 一致）
    # 所有分支都必须给 I_mag/soc_next/p_cell_applied 赋值，否则 return 时 UnboundLocalError（尤其零功率分支）
    p_cell_applied = 0.0
    soc_next = soc0
    I_mag = 0.0

    if abs(p_env_applied) < 1e-9:
        pass   # 已用上面默认值

    elif p_env_applied < 0:  # discharge
        I_soc_lim_raw = max(0.0, (soc0 - cell.soc_min) * Qa_Ah * 3600.0 / dt_s)

        Vmin_safe = 0.1
        I_voltage_lim = max(0.0, (Voc0 - Vmin_safe) / R_overall) if R_overall > 1e-9 else 0.0
        I_soc_lim = min(I_soc_lim_raw, I_voltage_lim)

        if I_soc_lim > 0:
            V_at_I = Voc0 - I_soc_lim * R_overall
            P_soc_lim = max(0.0, I_soc_lim * V_at_I)
        else:
            P_soc_lim = 0.0

        P_allowed_mag = min(abs(p_env_applied), P_soc_lim)
        p_cell_applied = -P_allowed_mag

        # 反算放电电流幅值
        from cell_slb import current_from_power_discharge
        I_mag = current_from_power_discharge(P_allowed_mag, Voc0, R_overall)
        I_signed = +I_mag
        soc_next = soc0 - (I_signed * dt_s) / (Qa_Ah * 3600.0)

    else:  # charge
        I_soc_lim = max(0.0, (cell.soc_max - soc0) * Qa_Ah * 3600.0 / dt_s)
        P_soc_lim = max(0.0, I_soc_lim * (Voc0 + I_soc_lim * R_overall))

        P_allowed = min(p_env_applied, P_soc_lim)
        p_cell_applied = +P_allowed

        from cell_slb import current_from_power_charge
        I_mag = current_from_power_charge(P_allowed, Voc0, R_overall)
        I_signed = -I_mag
        soc_next = soc0 - (I_signed * dt_s) / (Qa_Ah * 3600.0)

    soc_next = float(np.clip(soc_next, cell.soc_min, cell.soc_max))

    # cell(W) -> pack(W) -> system(kW)
    p_pack_applied_w = p_cell_applied * n_cells
    p_sys_applied_kw = (p_pack_applied_w * system.n_packs) / 1000.0
    return soc_next, float(p_sys_applied_kw), float(I_mag)


def _precompute_one_soc_row_cell_efc(j, soc, system, actions, soh_fixed, dt_s, soc_grid, Q_cell):
    """
    计算一个 SOC 网格点 j 的整行查表：
      next_idx[j,:], p_applied[j,:], delta_efc[j,:]
    delta_efc 用 cell-level：(|I_cell|*dt_h)/(2*Q_cell)
    """
    Nact = len(actions)
    next_row = np.empty(Nact, dtype=np.int16 if len(soc_grid) < 32767 else np.int32)
    papp_row = np.empty(Nact, dtype=np.float32)
    efc_row = np.empty(Nact, dtype=np.float32)

    dt_h = dt_s / 3600.0

    for a, p_req in enumerate(actions):
        soc_next, p_app, I_cell_mag = transition_soc_ah_fixed_soh(
            system, float(soc), float(p_req), soh_fixed, dt_s=dt_s
        )

        # 向下取整到网格，避免顶格处小幅放电被舍入回同一格
        idx = np.searchsorted(soc_grid, float(soc_next), side="right") - 1
        k = int(np.clip(idx, 0, len(soc_grid) - 1))
        next_row[a] = k
        papp_row[a] = float(p_app)

        # cell-level EFC increment
        efc_row[a] = float(I_cell_mag) * dt_h / (2.0 * Q_cell)

    return j, next_row, papp_row, efc_row

def precompute_tables_parallel_cell_efc(system, soc_grid, actions, soh_fixed, dt_s=3600.0, n_jobs=-1, backend="loky"):
    """
    多进程并行预计算（按 SOC 行并行）。
    actions: 1D (Nact,) 则所有 SOC 共用；2D (Nsoc, Nact) 则每行 j 用 actions[j, :]。
    返回：next_idx, p_applied, delta_efc（均为 [Nsoc, Nact]）
    """
    Nsoc = len(soc_grid)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions_per_j = [actions for _ in range(Nsoc)]
    else:
        assert actions.shape[0] == Nsoc
        actions_per_j = [actions[j, :] for j in range(Nsoc)]
    Nact = len(actions_per_j[0])
    Q_cell = float(system.pack.cell.nominal_capacity_Ah)

    results = Parallel(n_jobs=n_jobs, backend=backend, batch_size=1)(
        delayed(_precompute_one_soc_row_cell_efc)(
            j, soc_grid[j], system, actions_per_j[j], soh_fixed, dt_s, soc_grid, Q_cell
        )
        for j in range(Nsoc)
    )

    next_idx = np.empty((Nsoc, Nact), dtype=np.int16 if Nsoc < 32767 else np.int32)
    p_applied = np.empty((Nsoc, Nact), dtype=np.float32)
    delta_efc = np.empty((Nsoc, Nact), dtype=np.float32)

    for j, next_row, papp_row, efc_row in results:
        next_idx[j, :] = next_row
        p_applied[j, :] = papp_row
        delta_efc[j, :] = efc_row

    return next_idx, p_applied, delta_efc

@njit(parallel=True, fastmath=True)
def dp_update_one_t_numba(
    V_next,        # (Nsoc,) float64
    next_idx,      # (Nsoc, Nact) int32
    p_applied,     # (Nsoc, Nact) float64
    delta_efc,     # (Nsoc, Nact) float64
    nl_t,          # float64
    p_t,           # float64
    dt_h,          # float64
    lam,           # float64
    allow_export   # bool
):
    Nsoc, Nact = next_idx.shape
    INF = 1e30
    V_row = np.empty(Nsoc, dtype=np.float64)
    pi_row = np.empty(Nsoc, dtype=np.int32)

    for j in prange(Nsoc):
        best = INF
        best_a = 0
        for a in range(Nact):
            k = next_idx[j, a]
            grid_p = nl_t + p_applied[j, a]
            if (not allow_export) and (grid_p < 0.0):
                continue

            # stage cost: grid + lam * delta_efc (no degradation term)
            cost = p_t * grid_p * dt_h + lam * delta_efc[j, a] + V_next[k]

            if cost < best:
                best = cost
                best_a = a

        V_row[j] = best
        pi_row[j] = best_a

    return V_row, pi_row

def solve_for_lambda_numba(
    test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
    dt_s=3600.0, terminal_equal_init=True, allow_export=True, lam=0.0
):
    df = test_df.reset_index(drop=True).copy()
    T = len(df)
    dt_h = dt_s / 3600.0

    price = df["Price"].to_numpy(np.float64)
    net_load = (df["Demand"].to_numpy(np.float64) - df["PV"].to_numpy(np.float64))

    Nsoc, Nact = next_idx.shape
    INF = 1e30

    soc0 = float(system.soc)
    j0 = int(np.argmin(np.abs(soc_grid - soc0)))

    # dtype 准备（Numba 需要稳定 dtype）
    next_i32 = next_idx.astype(np.int32, copy=False)
    papp_f64 = p_applied.astype(np.float64, copy=False)
    defc_f64 = delta_efc.astype(np.float64, copy=False)

    V_next = np.full(Nsoc, INF, dtype=np.float64)
    V_curr = np.full(Nsoc, INF, dtype=np.float64)
    pi = np.full((T, Nsoc), -1, dtype=np.int16 if Nact < 32767 else np.int32)

    if terminal_equal_init:
        V_next[:] = INF
        V_next[j0] = 0.0
    else:
        V_next[:] = 0.0

    # backward DP with Numba
    for t in range(T - 1, -1, -1):
        V_row, pi_row = dp_update_one_t_numba(
            V_next, next_i32, papp_f64, defc_f64,
            float(net_load[t]), float(price[t]),
            float(dt_h), float(lam),
            bool(allow_export)
        )
        V_next[:] = V_row
        pi[t, :] = pi_row.astype(pi.dtype, copy=False)

    # forward rollout (same as before)
    j = j0
    soc_path = np.empty(T, dtype=np.float32)
    p_app_path = np.empty(T, dtype=np.float32)
    grid_path = np.empty(T, dtype=np.float64)
    efc_cum = np.empty(T, dtype=np.float64)

    efc_total = 0.0
    for t in range(T):
        soc_path[t] = soc_grid[j]
        a = int(pi[t, j])

        p_app = float(p_applied[j, a])
        p_app_path[t] = p_app

        grid_p = float(net_load[t] + p_app)
        if not allow_export:
            grid_p = max(0.0, grid_p)
        grid_path[t] = grid_p

        efc_total += float(delta_efc[j, a])
        efc_cum[t] = efc_total

        j = int(next_idx[j, a])

    grid_cost = price * grid_path * dt_h
    total_grid_cost = float(np.sum(grid_cost))

    out = df.copy()
    out["SOC_opt"] = soc_path
    out["P_applied_opt_kW"] = p_app_path
    out["Grid_P_opt_kW"] = grid_path
    out["Cost_grid_opt"] = grid_cost
    out["EFC_cum_cell"] = efc_cum
    out["lambda"] = float(lam)

    return efc_total, total_grid_cost, out

def bisect_lambda_for_efc_budget_vectorized(
    test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
    target_efc=300.0, dt_s=3600.0,
    terminal_equal_init=True, allow_export=True,
    max_iter=25, tol=1e-3
):
    # λ=0
    efc0, cost0, out0 = solve_for_lambda_numba(
        test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
        dt_s=dt_s, terminal_equal_init=terminal_equal_init, allow_export=allow_export, lam=0.0
    )
    if efc0 <= target_efc + tol:
        return dict(lam=0.0, efc=efc0, cost=cost0, out=out0)

    # find hi
    lam_lo, lam_hi = 0.0, 1.0
    best = None
    for _ in range(40):
        efc_hi, cost_hi, out_hi = solve_for_lambda_numba(
            test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
            dt_s=dt_s, terminal_equal_init=terminal_equal_init, allow_export=allow_export, lam=lam_hi
        )
        if efc_hi <= target_efc + tol:
            best = dict(lam=lam_hi, efc=efc_hi, cost=cost_hi, out=out_hi)
            break
        lam_hi *= 2.0

    if best is None:
        return dict(lam=lam_hi, efc=efc_hi, cost=cost_hi, out=out_hi,
                    note="Could not meet EFC budget even with very large lambda (check constraints/feasibility).")

    # bisection
    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        efc_mid, cost_mid, out_mid = solve_for_lambda_numba(
            test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
            dt_s=dt_s, terminal_equal_init=terminal_equal_init, allow_export=allow_export, lam=lam_mid
        )

        if efc_mid <= target_efc + tol:
            best = dict(lam=lam_mid, efc=efc_mid, cost=cost_mid, out=out_mid)
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid

        if abs(efc_mid - target_efc) < 0.05:
            break

    return best

def step_c_optionB_fast_parallel(
    test_df, system, dt_s=3600.0,
    soc_grid_size=101, action_grid_size=81,
    terminal_equal_init=True, allow_export=True,
    target_efc=300.0,
    n_jobs=-1
):
    cell = system.pack.cell
    soh_fixed = float(get_fixed_soh(cell))  # 你已有

    soc_grid = np.linspace(float(cell.soc_min), float(cell.soc_max), soc_grid_size).astype(np.float32)
    actions = build_soc_dependent_actions(system, soc_grid, action_grid_size=action_grid_size, strict_boundary=True)

    # 并行预计算（cell-level EFC）
    next_idx, p_applied, delta_efc = precompute_tables_parallel_cell_efc(
        system, soc_grid, actions, soh_fixed, dt_s=dt_s, n_jobs=n_jobs, backend="loky"
    )

    # λ 二分（矢量 DP）
    best = bisect_lambda_for_efc_budget_vectorized(
        test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
        target_efc=target_efc, dt_s=dt_s,
        terminal_equal_init=terminal_equal_init, allow_export=allow_export
    )

    # baseline cost
    dt_h = dt_s / 3600.0
    net_load = (test_df["Demand"].to_numpy(np.float64) - test_df["PV"].to_numpy(np.float64))
    price = test_df["Price"].to_numpy(np.float64)
    baseline_cost = float(np.sum(price * net_load * dt_h))  # allow_export=True 时可为负/正

    summary = dict(
        lambda_star=float(best["lam"]),
        efc_used_cell=float(best["efc"]),
        optimal_grid_cost=float(best["cost"]),
        baseline_cost=float(baseline_cost),
        max_savings=float(baseline_cost - best["cost"]),
        target_efc=float(target_efc),
        soc_grid_size=int(soc_grid_size),
        action_grid_size=int(action_grid_size),
        terminal_equal_init=bool(terminal_equal_init),
        allow_export=bool(allow_export),
        SOH_fixed=float(soh_fixed),
        n_jobs=int(n_jobs) if n_jobs != -1 else -1,
    )
    return best["out"], summary


def to_hourly_df_for_plotting(out, dt_s=3600, time_col="datetime_utc"):
    """
    把 DP/RL/任意控制结果 out 转成 plot_battery_performance.py 可直接画图的格式：
    - Demand, PV 必须存在（你的 test_df 已有）
    - p_grid_kw, p_batt_applied_kw, soc 必须存在
    - 额外补充 grid_cost, grid_cost_cum, grid_import_kwh, grid_export_kwh 便于其它图
    """
    df = out.copy()

    # 1) 统一时间列 dtype
    if time_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # 2) 统一 SOC/功率列名（适配 pbp）
    # 你的 DP 输出列名可能不同，这里做兼容
    if "soc" not in df.columns:
        if "SOC_opt" in df.columns:
            df["soc"] = df["SOC_opt"]
        elif "SOC" in df.columns:
            df["soc"] = df["SOC"]
        else:
            raise ValueError("Missing SOC column: expected 'soc' or 'SOC_opt' or 'SOC'.")

    if "p_batt_applied_kw" not in df.columns:
        if "P_applied_opt_kW" in df.columns:
            df["p_batt_applied_kw"] = df["P_applied_opt_kW"]
        elif "p_batt_kw" in df.columns:
            df["p_batt_applied_kw"] = df["p_batt_kw"]
        else:
            raise ValueError("Missing battery power column: expected 'p_batt_applied_kw' or 'P_applied_opt_kW'.")

    if "p_grid_kw" not in df.columns:
        if "Grid_P_opt_kW" in df.columns:
            df["p_grid_kw"] = df["Grid_P_opt_kW"]
        else:
            # 如果没存 grid，按功率平衡补出来：grid = demand - pv + batt
            if not {"Demand", "PV"}.issubset(df.columns):
                raise ValueError("Missing 'Demand'/'PV' to compute grid power.")
            df["p_grid_kw"] = df["Demand"].astype(float) - df["PV"].astype(float) + df["p_batt_applied_kw"].astype(float)

    # 3) 计算电量与成本（供其它图使用）
    dt_h = dt_s / 3600.0
    price = df["Price"].astype(float).to_numpy()

    p_grid = df["p_grid_kw"].astype(float).to_numpy()
    import_kwh = np.maximum(p_grid, 0.0) * dt_h
    export_kwh = np.maximum(-p_grid, 0.0) * dt_h

    df["grid_import_kwh"] = import_kwh
    df["grid_export_kwh"] = export_kwh

    # buy price = sell price（你设定）
    # 成本 = price * (import - export) = price * p_grid * dt_h
    df["grid_cost"] = price * p_grid * dt_h
    df["grid_cost_cum"] = np.cumsum(df["grid_cost"].to_numpy())

    return df





# NEW UPDATE !!!.
# ======================================================

"""
Piecewise-lambda (Scheme B) implementation for 5y theoretical-limit DP.

Goal:
- Split the 5-year horizon into blocks (e.g. every 3 months).
- Allocate an EFC budget to each block (based on price-volatility opportunity weights).
- For each block, solve a *single* lambda via your existing DP+bisect so that
  EFC_block(lambda) ≈ EFC_budget_block.
- Enforce SOC continuity across blocks (carry end-SOC to next block).
- Use an outer bisection on a global scale factor alpha to make total EFC ≈ 300 (and <= 300).

This code assumes you already have in your tlc file:
- build_system
- get_fixed_soh
- build_soc_dependent_actions
- precompute_tables_parallel_cell_efc
- bisect_lambda_for_efc_budget_vectorized
- solve_for_lambda_numba  (used internally by bisect)
- step_c_optionB_fast_parallel (not used here; we reuse its internals but precompute once)
"""



# ----------------------------
# 1) Utilities: split blocks + weights
# ----------------------------
def split_into_blocks(df: pd.DataFrame, freq: str = "3MS", time_col: str = "datetime_utc"):
    """
    Split df into blocks by pandas Grouper frequency.
    freq examples:
      - "3MS" : every 3 months anchored at month-start
      - "QS"  : quarter-start
      - "MS"  : month-start
    Returns: list of (block_key, df_block)
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    dff = df.copy()
    dff[time_col] = pd.to_datetime(dff[time_col])
    dff = dff.sort_values(time_col).reset_index(drop=True)

    blocks = []
    for key, g in dff.groupby(pd.Grouper(key=time_col, freq=freq)):
        if len(g) == 0:
            continue
        blocks.append((key, g.reset_index(drop=True)))
    return blocks


def compute_block_weights(
    df_block: pd.DataFrame,
    allow_export: bool = True,
    use_abs_net_load_if_export: bool = True,
):
    """
    Opportunity weight w_k for EFC allocation.

    A simple, robust choice:
      w = (p90(price)-p10(price)) * mean(|net_load|)   if allow_export
      w = (p90(price)-p10(price)) * mean(max(net_load,0)) if not allow_export

    If allow_export=False, negative grid is clipped, so only positive net-load is "useful".
    """
    price = df_block["Price"].to_numpy(np.float64)
    net_load = (df_block["Demand"].to_numpy(np.float64) - df_block["PV"].to_numpy(np.float64))

    spread = float(np.percentile(price, 90) - np.percentile(price, 10))
    if spread < 1e-9:
        spread = 1e-9

    if allow_export:
        if use_abs_net_load_if_export:
            amp = float(np.mean(np.abs(net_load)))
        else:
            # alternative: only positive import side
            amp = float(np.mean(np.maximum(net_load, 0.0)))
    else:
        amp = float(np.mean(np.maximum(net_load, 0.0)))

    amp = max(amp, 1e-9)
    return spread * amp


def allocate_budgets(weights: np.ndarray, efc_total: float, alpha: float):
    """
    budgets_k = alpha * efc_total * w_k / sum(w)
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    budgets = alpha * efc_total * (w / np.sum(w))
    return budgets


# ----------------------------
# 2) One-pass piecewise solve (given budgets per block)
# ----------------------------
def run_piecewise_lambda_once(
    blocks,
    system,
    soc_grid,
    actions,
    next_idx,
    p_applied,
    delta_efc,
    budgets,
    dt_s: float = 3600.0,
    terminal_equal_init: bool = False,
    allow_export: bool = True,
    per_block_tol: float = 1e-3,
    n_jobs: int = -1,  # kept for compatibility; precompute already done
):
    """
    Run piecewise lambda with given per-block EFC budgets.
    Returns:
      out_full_df, block_summaries(list of dict), total_efc, total_cost_opt, total_cost_baseline
    """
    assert len(blocks) == len(budgets)

    dt_h = dt_s / 3600.0

    out_list = []
    summaries = []

    total_efc = 0.0
    total_cost_opt = 0.0
    total_cost_baseline = 0.0

    # SOC continuity: initialize from current system.soc
    soc_init = float(system.soc)

    for k, ((block_key, df_block), efc_budget_k) in enumerate(zip(blocks, budgets), start=1):
        # set SOC for this block
        system.pack.cell.soc = soc_init

        # solve lambda for this block budget
        best = bisect_lambda_for_efc_budget_vectorized(
            df_block, system, soc_grid, actions, next_idx, p_applied, delta_efc,
            target_efc=float(efc_budget_k), dt_s=dt_s,
            terminal_equal_init=terminal_equal_init,
            allow_export=allow_export,
            max_iter=25, tol=per_block_tol
        )

        out_k = best["out"].copy()
        out_k["block_id"] = k
        out_k["block_key"] = str(block_key)
        out_k["block_efc_budget"] = float(efc_budget_k)
        out_k["block_lambda_star"] = float(best["lam"])

        # block EFC used (within this block)
        efc_k = float(best["efc"])
        # end SOC (carry)
        soc_end = float(out_k["SOC_opt"].iloc[-1])
        soc_init = soc_end

        # costs
        # baseline cost under the same export assumption
        price = df_block["Price"].to_numpy(np.float64)
        net_load = (df_block["Demand"].to_numpy(np.float64) - df_block["PV"].to_numpy(np.float64))
        if not allow_export:
            net_load = np.maximum(net_load, 0.0)
        baseline_cost_k = float(np.sum(price * net_load * dt_h))
        opt_cost_k = float(best["cost"])

        total_efc += efc_k
        total_cost_opt += opt_cost_k
        total_cost_baseline += baseline_cost_k

        summaries.append(dict(
            block_id=k,
            block_key=str(block_key),
            n_steps=int(len(df_block)),
            efc_budget=float(efc_budget_k),
            efc_used=float(efc_k),
            lambda_star=float(best["lam"]),
            soc_start=float(out_k["SOC_opt"].iloc[0]),
            soc_end=float(soc_end),
            baseline_cost=float(baseline_cost_k),
            optimal_cost=float(opt_cost_k),
            savings=float(baseline_cost_k - opt_cost_k),
        ))

        out_list.append(out_k)

    out_full = pd.concat(out_list, axis=0, ignore_index=True)
    return out_full, summaries, float(total_efc), float(total_cost_opt), float(total_cost_baseline)


# ----------------------------
# 3) Outer alpha bisection to hit total EFC ~ efc_total (and <= efc_total)
# ----------------------------
def run_piecewise_lambda_with_alpha_bisect(
    df_5y: pd.DataFrame,
    system,
    efc_total: float = 300.0,
    block_freq: str = "3MS",
    dt_s: float = 3600.0,
    soc_grid_size: int = 101,
    action_grid_size: int = 401,   # IMPORTANT: do NOT use 20001 here, too slow for piecewise
    terminal_equal_init: bool = False,
    allow_export: bool = True,
    n_jobs: int = -1,
    time_col: str = "datetime_utc",
    total_tol: float = 0.5,        # tolerance for hitting total EFC
    max_alpha_iter: int = 12,
    per_block_tol: float = 1e-3,
):
    """
    Full pipeline:
    - fixed SOH (as in your Step C)
    - precompute DP tables ONCE (soc_grid, actions, next_idx, p_applied, delta_efc)
    - compute block weights -> budgets(alpha)
    - outer bisection on alpha so total EFC ~= efc_total and <= efc_total
    """
    # --- Build grids & precompute ONCE ---
    cell = system.pack.cell
    soh_fixed = float(get_fixed_soh(cell))

    soc_grid = np.linspace(float(cell.soc_min), float(cell.soc_max), soc_grid_size).astype(np.float32)
    actions = build_soc_dependent_actions(system, soc_grid, action_grid_size=action_grid_size, strict_boundary=True)

    next_idx, p_applied, delta_efc = precompute_tables_parallel_cell_efc(
        system, soc_grid, actions, soh_fixed, dt_s=dt_s, n_jobs=n_jobs, backend="loky"
    )

    # --- Split into blocks ---
    blocks = split_into_blocks(df_5y, freq=block_freq, time_col=time_col)
    if len(blocks) < 2:
        raise ValueError(f"Not enough blocks from freq={block_freq}. Check datetime range / freq.")

    # --- Weights ---
    weights = np.array([compute_block_weights(g, allow_export=allow_export) for _, g in blocks], dtype=np.float64)

    # --- Alpha bracket ---
    # We want total_efc(alpha) to be monotonic increasing in alpha (usually true).
    # We'll search alpha in [lo, hi] s.t.:
    #   total_efc(lo) <= efc_total
    #   total_efc(hi) >= efc_total  (or saturates)
    alpha_lo = 0.0
    alpha_hi = 1.0

    # helper: budgets with "last block = remaining" so total budget exactly = alpha*efc_total (but capped at remaining)
    def make_budgets(alpha):
        b = allocate_budgets(weights, efc_total=efc_total, alpha=alpha)
        # enforce exact total budget = alpha*efc_total by residual correction in last block
        # (this is budget accounting; actual used EFC can still differ)
        total_b = float(np.sum(b))
        if total_b <= 1e-12:
            b[:] = 0.0
        else:
            # make it exact
            b = b * (alpha * efc_total / total_b)
        return b

    # evaluate alpha_hi, increase until we either exceed efc_total or we see saturation
    last_efc = None
    cache_hi = None
    for _ in range(6):
        budgets_hi = make_budgets(alpha_hi)
        out_hi, sums_hi, efc_hi, cost_hi, base_hi = run_piecewise_lambda_once(
            blocks, system, soc_grid, actions, next_idx, p_applied, delta_efc,
            budgets_hi, dt_s=dt_s,
            terminal_equal_init=terminal_equal_init,
            allow_export=allow_export,
            per_block_tol=per_block_tol,
            n_jobs=n_jobs,
        )
        cache_hi = (out_hi, sums_hi, efc_hi, cost_hi, base_hi, budgets_hi, alpha_hi)
        if efc_hi >= efc_total - total_tol:
            break
        # saturation check: if increasing alpha doesn't change efc much, stop expanding
        if last_efc is not None and abs(efc_hi - last_efc) < 0.05:
            break
        last_efc = efc_hi
        alpha_hi *= 2.0

    # evaluate alpha_lo (0 gives zero budgets => should yield ~0 efc)
    budgets_lo = make_budgets(alpha_lo)
    out_lo, sums_lo, efc_lo, cost_lo, base_lo = run_piecewise_lambda_once(
        blocks, system, soc_grid, actions, next_idx, p_applied, delta_efc,
        budgets_lo, dt_s=dt_s,
        terminal_equal_init=terminal_equal_init,
        allow_export=allow_export,
        per_block_tol=per_block_tol,
        n_jobs=n_jobs,
    )
    best_feasible = (out_lo, sums_lo, efc_lo, cost_lo, base_lo, budgets_lo, alpha_lo, soh_fixed)

    # if even alpha_hi doesn't get close, return hi result (means physical/max usage < target)
    if cache_hi is not None and cache_hi[2] < efc_total - total_tol:
        out_hi, sums_hi, efc_hi, cost_hi, base_hi, budgets_hi, alpha_used = cache_hi[0], cache_hi[1], cache_hi[2], cache_hi[3], cache_hi[4], cache_hi[5], cache_hi[6]
        return dict(
            out=out_hi,
            block_summaries=sums_hi,
            total_efc=efc_hi,
            total_cost_opt=cost_hi,
            total_cost_baseline=base_hi,
            alpha=alpha_used,
            budgets=budgets_hi,
            weights=weights,
            note="Total EFC saturates below target_efc_total (even with very large alpha).",
            SOH_fixed=float(soh_fixed),
        )

    # --- Outer bisection ---
    lo, hi = alpha_lo, cache_hi[6] if cache_hi is not None else alpha_hi
    for _ in range(max_alpha_iter):
        mid = 0.5 * (lo + hi)
        budgets_mid = make_budgets(mid)

        out_mid, sums_mid, efc_mid, cost_mid, base_mid = run_piecewise_lambda_once(
            blocks, system, soc_grid, actions, next_idx, p_applied, delta_efc,
            budgets_mid, dt_s=dt_s,
            terminal_equal_init=terminal_equal_init,
            allow_export=allow_export,
            per_block_tol=per_block_tol,
            n_jobs=n_jobs,
        )

        # keep best feasible (<= efc_total and closest)
        if efc_mid <= efc_total + 1e-9:
            if (efc_total - efc_mid) < (efc_total - best_feasible[2]):
                best_feasible = (out_mid, sums_mid, efc_mid, cost_mid, base_mid, budgets_mid, mid, soh_fixed)
            lo = mid
        else:
            hi = mid

        if abs(efc_mid - efc_total) <= total_tol and efc_mid <= efc_total + 1e-9:
            best_feasible = (out_mid, sums_mid, efc_mid, cost_mid, base_mid, budgets_mid, mid, soh_fixed)
            break

    out_best, sums_best, efc_best, cost_best, base_best, budgets_best, alpha_best, soh_fixed = best_feasible

    return dict(
        out=out_best,
        block_summaries=sums_best,
        total_efc=float(efc_best),
        total_cost_opt=float(cost_best),
        total_cost_baseline=float(base_best),
        alpha=float(alpha_best),
        budgets=np.asarray(budgets_best, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        note="OK",
        SOH_fixed=float(soh_fixed),
        soc_grid_size=int(soc_grid_size),
        action_grid_size=int(action_grid_size),
        block_freq=str(block_freq),
        allow_export=bool(allow_export),
        terminal_equal_init=bool(terminal_equal_init),
    )


# ----------------------------
# 4) Example usage (match your calling style)
# ----------------------------
if __name__ == "__main__":
    # Example: Build system
    system_kwargs = dict(
        cache_dir="./battery_model_params_20260118/1_hour_0.0_1.0",
        Ns=96,
        Np=10,
        N_series=210,
        N_parallel=1,
    )
    system = build_system(**system_kwargs)

    # Load your df_5y as test_df (must include datetime_utc, Demand, PV, Price)
    # test_df = ...

    result = run_piecewise_lambda_with_alpha_bisect(
        df_5y=test_df,
        system=system,
        efc_total=300.0,
        block_freq="3MS",       # every 3 months
        dt_s=3600.0,
        soc_grid_size=101,
        action_grid_size=401,   # start smaller; increase if needed
        terminal_equal_init=False,
        allow_export=True,
        n_jobs=-1,
        time_col="datetime_utc",
        total_tol=0.5,
        max_alpha_iter=12,
        per_block_tol=1e-3,
    )

    out_full = result["out"]
    block_summ = pd.DataFrame(result["block_summaries"])
    print("alpha =", result["alpha"])
    print("total_efc =", result["total_efc"])
    print("baseline_cost =", result["total_cost_baseline"])
    print("optimal_cost  =", result["total_cost_opt"])
    print("saving =", result["total_cost_baseline"] - result["total_cost_opt"])
    print(block_summ.head())

