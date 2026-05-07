# Author: Kaifeng ZHU
# This file contains the main functions of theo_dp_core.
import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange
from cell_slb import current_from_power_discharge
from theoratical_limit_calculation import (
    get_fixed_soh,
    build_global_actions,
    transition_soc_ah_fixed_soh,
)

def _precompute_one_soc_row_cell_efc(j, soc0, system, actions, soh_fixed, dt_s, soc_grid, Q_cell):
    Nact = len(actions)
    next_row = np.empty(Nact, dtype=np.int16 if len(soc_grid) < 32767 else np.int32)
    papp_row = np.empty(Nact, dtype=np.float32)
    efc_row  = np.empty(Nact, dtype=np.float32)

    dt_h = dt_s / 3600.0
    cell = system.pack.cell

    for a in range(Nact):
        p_req_sys_kw = float(actions[a])

        if abs(p_req_sys_kw) < 1e-12:
            soc_next = float(soc0)
            p_applied_sys_kw = 0.0
        else:
            soc_next, p_applied_sys_kw, _ = transition_soc_ah_fixed_soh(
                system, float(soc0), p_req_sys_kw, soh_fixed, dt_s=dt_s
            )

        # 映射到 SOC 网格：向下取整，避免顶格(0.8)处小幅放电被“四舍五入”回同一格导致 DP 永远不离开
        soc_next_f = float(soc_next)
        idx = np.searchsorted(soc_grid, soc_next_f, side="right") - 1
        k = int(np.clip(idx, 0, len(soc_grid) - 1))
        next_row[a] = k
        papp_row[a] = float(p_applied_sys_kw)

        # cell-level EFC increment (与你 theo 文件一致：I_mag*dt/(2Q))
        # 把 applied 系统功率转换成 cell 功率，估算 I
        p_app_sys_w = float(p_applied_sys_kw) * 1000.0
        p_app_pack_w = p_app_sys_w / system.n_packs
        n_cells = system.pack.Ns * system.pack.Np
        p_app_cell_w = p_app_pack_w / n_cells

        if abs(p_app_cell_w) < 1e-9:
            efc_row[a] = 0.0
            continue

        Voc0 = cell.luts["Voc"](soh_fixed, float(soc0))
        Rint0 = cell.luts["Rint"](soh_fixed, float(soc0))
        R10 = cell.luts["R1"](soh_fixed, float(soc0))
        R20 = cell.luts["R2"](soh_fixed, float(soc0))
        R_overall = Rint0 + R10 + R20

        # 用 discharge 的公式求 I_mag（充电也用近似同量级，跟你 theo 文件口径保持一致即可）
        I_mag = current_from_power_discharge(abs(float(p_app_cell_w)), float(Voc0), float(R_overall))
        efc_row[a] = float(I_mag) * dt_h / (2.0 * float(Q_cell))

    return j, next_row, papp_row, efc_row

def precompute_tables_parallel_cell_efc(system, soc_grid, actions, soh_fixed, dt_s=3600.0, n_jobs=-1, backend="loky"):
    """
    actions: 1D (Nact,) 则所有 SOC 共用同一组动作；2D (Nsoc, Nact) 则每行 j 使用 actions[j, :]。
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
        delayed(_precompute_one_soc_row_cell_efc)(j, soc_grid[j], system, actions_per_j[j], soh_fixed, dt_s, soc_grid, Q_cell)
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
    V_next, next_idx, p_applied, delta_efc,
    nl_t, buy_t, sell_t, dt_h, lam, allow_export
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

            if grid_p >= 0.0:
                energy_cost = buy_t * grid_p * dt_h
            else:
                energy_cost = -sell_t * (-grid_p) * dt_h

            cost = energy_cost + lam * delta_efc[j, a] + V_next[k]

            if cost < best:
                best = cost
                best_a = a

        V_row[j] = best
        pi_row[j] = best_a
    return V_row, pi_row

def solve_for_lambda_numba(test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
                          dt_s=3600.0, terminal_equal_init=True, allow_export=True, lam=0.0):
    df = test_df.reset_index(drop=True).copy()
    T = len(df)
    dt_h = dt_s / 3600.0
    buy_price = df["Buy_Price"].to_numpy(np.float64)
    sell_price = df["Sell_Price"].to_numpy(np.float64)
    net_load = (df["Demand"].to_numpy(np.float64) - df["PV"].to_numpy(np.float64))

    Nsoc, Nact = next_idx.shape
    INF = 1e30

    soc0 = float(system.soc)
    j0 = int(np.argmin(np.abs(soc_grid - soc0)))

    next_i32 = next_idx.astype(np.int32, copy=False)
    papp_f64 = p_applied.astype(np.float64, copy=False)
    defc_f64 = delta_efc.astype(np.float64, copy=False)

    V_next = np.full(Nsoc, INF, dtype=np.float64)
    pi = np.full((T, Nsoc), -1, dtype=np.int16 if Nact < 32767 else np.int32)

    if terminal_equal_init:
        V_next[:] = INF
        V_next[j0] = 0.0
    else:
        V_next[:] = 0.0

    for t in range(T - 1, -1, -1):
        V_row, pi_row = dp_update_one_t_numba(
            V_next, next_i32, papp_f64, defc_f64,
            float(net_load[t]),
            float(buy_price[t]),
            float(sell_price[t]),
            float(dt_h), float(lam), bool(allow_export)
        )

        V_next[:] = V_row
        pi[t, :] = pi_row.astype(pi.dtype, copy=False)

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

    grid_cost = np.where(
        grid_path >= 0.0,
        buy_price * grid_path * dt_h,
        -sell_price * (-grid_path) * dt_h
    )
    total_grid_cost = float(np.sum(grid_cost))

    out = df.copy()
    out["SOC_opt"] = soc_path
    out["P_applied_opt_kW"] = p_app_path
    out["Grid_P_opt_kW"] = grid_path
    out["Cost_grid_opt"] = grid_cost
    out["EFC_cum_cell"] = efc_cum
    out["lambda"] = float(lam)
    return efc_total, total_grid_cost, out

def bisect_lambda_for_efc_budget_vectorized(test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
                                           target_efc=300.0, dt_s=3600.0, terminal_equal_init=True,
                                           allow_export=True, max_iter=25, tol=1e-3):
    efc0, cost0, out0 = solve_for_lambda_numba(
        test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
        dt_s=dt_s, terminal_equal_init=terminal_equal_init, allow_export=allow_export, lam=0.0
    )
    if efc0 <= target_efc + tol:
        return dict(lam=0.0, efc=efc0, cost=cost0, out=out0)

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
                    note="Could not meet EFC budget even with very large lambda.")

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
    return best

def bisect_lambda_warm_start(
    test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
    target_efc,
    lam_init,
    lam_expand=1.5,
    dt_s=3600.0,
    terminal_equal_init=True,
    allow_export=True,
    max_iter=10,
    tol=1e-3,
):
    lam_lo = lam_init / lam_expand
    lam_hi = lam_init * lam_expand

    best = None

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        efc_mid, cost_mid, out_mid = solve_for_lambda_numba(
            test_df, system, soc_grid, actions, next_idx, p_applied, delta_efc,
            dt_s=dt_s,
            terminal_equal_init=terminal_equal_init,
            allow_export=allow_export,
            lam=lam_mid,
        )

        if efc_mid <= target_efc + tol:
            best = dict(lam=lam_mid, efc=efc_mid, cost=cost_mid, out=out_mid)
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid

    if best is None:
        best = dict(lam=lam_mid, efc=efc_mid, cost=cost_mid, out=out_mid)

    return best

