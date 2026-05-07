# Author: Kaifeng ZHU
# This file contains the main function of R-DA-MPC control simulation result generation.
import pandas as pd
import numpy as np

from theoratical_limit_calculation import build_system
from theo_dp_core_v2 import (
    precompute_tables_parallel_cell_efc,
    bisect_lambda_for_efc_budget_vectorized,
    solve_for_lambda_numba,
    bisect_lambda_warm_start
)
from theoratical_limit_calculation import get_fixed_soh, build_global_actions, build_soc_dependent_actions

# ========== 计算「下一天 action」时用到的数据（无校准时）==========
# Step A 求当日 λ（bisect）:
#   - df_day_dp: 当日 24h 的 Demand_Predicted, PV_Predicted, Price（仅预测）
#   - system.soc: 当日开始时的 SOC（来自昨日仿真的 step，依赖昨日真实 Demand/PV + 昨日 p_plan）
#   - system.pack.cell.efc_cycle: 当前累计 EFC（来自历史仿真）
#   - EFC_budget_day: (EFC_remaining / days_remaining) * (1 - safety_margin)
#   - lam_prev: 昨日得到的 λ（仅首日无；非首日用于 warm start 的 lam_init）
# Step B 用 λ 求当日 p_plan（solve_for_lambda_numba）:
#   - 同上 df_day_dp, system.soc, lam_star；DP 内部只用 test_df 的 Demand/PV/Price 和 lam
# 因此 λ 由「满足 EFC 预算」决定；若两 run 历史 EFC 用量相近则 λ 相同；λ 很大时最优策略≈少用电池，与 net_load 几乎无关。

def efc_budget_for_window(window_hours, life_years=5.0, efc_eol=300.0):
    life_hours = life_years * 365.0 * 24.0
    return (window_hours / life_hours) * efc_eol

def make_search_window_df(df, day_start, past_weeks=4, check_output=False):
    """
    window = [day_start - past_weeks*7天, day_start + 1天)，即 [past_start, day_end) 左闭右开。
    过去段（index < day_start）用真实 Demand/PV；未来一天（index >= day_start，含 day_start 当日 00:00~23:00）用 Predicted。
    注意：窗口内过去占绝大部分（如 past_weeks=1 时约 87.5% 为真实数据），
    校准得到的 λ 会强烈受真实负荷影响；零预测 run 若用此 λ，策略会趋近“少用电池”，
    可能与完美预测 run 表现相似。做零预测对比时建议设 calib_every_days=None 试跑。
    """
    past_start = day_start - pd.Timedelta(days=past_weeks * 7)
    day_end = day_start + pd.Timedelta(days=1)
    # 左闭右闭：从 past_start 到 day_start 当日最后一小时 (day_end - 1h)
    w = df.loc[past_start: day_end - pd.Timedelta(hours=1)].copy()

    # 关键：构造 DP 需要的 Demand/PV（历史用真实，未来一天用预测）
    # index >= day_start 的行（含 day_start 00:00）视为「未来」，用 Predicted
    is_future = w.index >= day_start
    w["Demand_dp"] = np.where(is_future, w["Demand_Predicted"].values, w["Demand"].values)
    w["PV_dp"] = np.where(is_future, w["PV_Predicted"].values, w["PV"].values)

    # DP core (theo_dp_core_v2) 需要 Buy_Price, Sell_Price；若 raw 只有 Price 则用同一列
    if "Buy_Price" in w.columns and "Sell_Price" in w.columns:
        out = w[["Demand_dp", "PV_dp", "Buy_Price", "Sell_Price"]].rename(
            columns={"Demand_dp": "Demand", "PV_dp": "PV"}
        )
    else:
        out = w[["Demand_dp", "PV_dp", "Price"]].rename(
            columns={"Demand_dp": "Demand", "PV_dp": "PV", "Price": "Buy_Price"}
        )
        out["Sell_Price"] = w["Price"].values

    if check_output:
        _check_search_window_out(out, w, day_start, is_future)
        # print(day_start)
        # if day_start == pd.Timestamp('2021-12-27'):
        #     print(out)
    return out


def _check_search_window_out(out, w, day_start, is_future):
    """校验 make_search_window_df 的 out：过去行=真实，未来行=预测；列名为 Demand, PV, Buy_Price, Sell_Price。"""
    assert list(out.columns) == ["Demand", "PV", "Buy_Price", "Sell_Price"], (
        f"out 列名应为 Demand, PV, Buy_Price, Sell_Price，得到 {list(out.columns)}"
    )
    past_mask = ~is_future
    if past_mask.any():
        np.testing.assert_array_almost_equal(
            out.loc[past_mask, "Demand"].values, w.loc[past_mask, "Demand"].values,
            err_msg="过去段 out['Demand'] 应等于 w['Demand']（真实）"
        )
        np.testing.assert_array_almost_equal(
            out.loc[past_mask, "PV"].values, w.loc[past_mask, "PV"].values,
            err_msg="过去段 out['PV'] 应等于 w['PV']（真实）"
        )
    if is_future.any():
        np.testing.assert_array_almost_equal(
            out.loc[is_future, "Demand"].values, w.loc[is_future, "Demand_Predicted"].values,
            err_msg="未来段 out['Demand'] 应等于 w['Demand_Predicted']"
        )
        np.testing.assert_array_almost_equal(
            out.loc[is_future, "PV"].values, w.loc[is_future, "PV_Predicted"].values,
            err_msg="未来段 out['PV'] 应等于 w['PV_Predicted']"
        )

def make_future_day_dp_df(df, day_start):
    day_end = day_start + pd.Timedelta(days=1)
    d = df.loc[day_start: day_end - pd.Timedelta(hours=1)].copy()
    if "Buy_Price" in d.columns and "Sell_Price" in d.columns:
        out = d[["Demand_Predicted", "PV_Predicted", "Buy_Price", "Sell_Price"]].rename(
            columns={"Demand_Predicted": "Demand", "PV_Predicted": "PV"}
        )
    else:
        out = d[["Demand_Predicted", "PV_Predicted", "Price"]].rename(
            columns={"Demand_Predicted": "Demand", "PV_Predicted": "PV"}
        )
        out["Buy_Price"] = out["Price"]
        out["Sell_Price"] = out["Price"]
        out = out[["Demand", "PV", "Buy_Price", "Sell_Price"]]
    return out

def simulate_one_day_with_real_system(system, df_day_real, p_batt_request_kw, dt_s=3600.0):
    dt_h = dt_s / 3600.0
    out = df_day_real.copy()
    n = len(out)

    p_app = np.zeros(n)
    soc = np.zeros(n)
    p_grid = np.zeros(n)
    cost = np.zeros(n)
    env_power_limit = np.zeros(n)           # 与请求同方向的 envelope 上限（兼容旧列）
    env_power_limit_charge = np.zeros(n)    # 当前 SOC 下充电方向 envelope 上限 (kW)
    env_power_limit_discharge = np.zeros(n) # 当前 SOC 下放电方向 envelope 上限 (kW, 正数)
    soc_power_limit = np.zeros(n)
    efc = np.zeros(n)

    for i in range(n):
        d = float(out["Demand"].iloc[i])
        pv = float(out["PV"].iloc[i])
        buy_price = float(out["Buy_Price"].iloc[i])
        sell_price = float(out["Sell_Price"].iloc[i])

        # 本步开始时的 SOC 下，充电/放电两个方向的 envelope 上限（便于诊断）
        env_power_limit_charge[i] = float(system.get_step_power_limit_kw("charge"))
        env_power_limit_discharge[i] = float(system.get_step_power_limit_kw("discharge"))

        step_out = system.step(float(p_batt_request_kw[i]), dt_s=dt_s)
        p_app[i] = float(step_out["p_applied_sys_kw"])
        soc[i] = float(step_out["soc_after"])
        env_power_limit[i] = float(step_out["p_envelope_limit_sys_kw"])  # 与请求同向
        soc_power_limit[i] = float(step_out["p_soc_limit_sys_kw"])

        # 电网功率 (kW)：正=从电网取电，负=向电网送电；成本 = 电价 × 功率 × 时长 (price·kW·h = 金额)
        pg = d - pv + p_app[i]
        p_grid[i] = pg
        if pg >= 0.0:
            cost[i] = buy_price * pg * dt_h
        else:
            cost[i] = -sell_price * (-pg) * dt_h

        # 记录每个时间步后的累计等效全循环次数（cell 级）
        efc[i] = float(system.pack.cell.efc_cycle)

    out["p_batt_request_kw"] = p_batt_request_kw
    out["p_batt_applied_kw"] = p_app
    out["env_power_limit_kw"] = env_power_limit
    out["env_power_limit_charge_kw"] = env_power_limit_charge
    out["env_power_limit_discharge_kw"] = env_power_limit_discharge
    out["soc_power_limit_kw"] = soc_power_limit
    out["soc"] = soc
    out["p_grid_kw"] = p_grid
    out["grid_cost"] = cost
    out["grid_cost_cum"] = np.cumsum(cost)
    out["soh"] = float(system.pack.cell.soh)
    # 改为逐时记录累计 EFC，而不是整天一个常数
    out["efc_cycle"] = efc
    return out

def run_rolling_5y(
    df_raw,  # 必须含列: datetime_utc, Demand, PV, Demand_Predicted, PV_Predicted, Price
    cache_dir,
    Ns, Np, N_series, N_parallel,
    start="2020-12-27",
    end="2025-12-26",
    calib_every_days = 7,
    past_weeks=4,
    dt_s=3600.0,
    soc_grid_size=101,
    action_grid_size=81,
    EFC_EOL = 300.0,
    safety_margin = 0.05,
    lam_expand_day = 1.5,
    lam_expand_calib = 2.0,
    max_iter_day = 8,
    max_iter_calib = 10,
    allow_export=True,
    terminal_equal_init=True,
    n_jobs=-1,
    debug_no_future_leak=False,  # 若 True：首日检查 DP 输入仅来自 Predicted，与 Actual 不同则断言
    check_search_window=False,   # 若 True：校准时对 make_search_window_df 的 out 做自检（过去=真实，未来=预测）
    use_calibrated_lambda_only=False,  # 若 True：禁用“每日 bisect 求 λ”，仅在校准日更新 λ，其余日直接用上次校准的 λ 生成次日策略
    return_calibration_log=False,      # 若 True：同时返回 (res, calib_log_df)
):
    """
    滚动仿真：每日用 Demand_Predicted/PV_Predicted 做 DP 求 p_plan，再用真实 Demand/PV 做 step 算成本。
    优化阶段绝不使用当日真实 Demand/PV，无未来信息泄露。

    若「零预测」与「完美预测」表现相同，可能原因与排查：
    1) 校准 λ 被历史真实数据主导：每周校准用 make_search_window_df = 过去几周(真实) + 当日(预测)。
       零预测时当日为 0，但窗口大部分是真实负荷，校准出的 λ 与完美预测时接近；
       λ 很大时两 run 都倾向少用电池 → p_plan 都≈0 → 仿真成本相同。
       排查：设 calib_every_days=None 关掉校准再跑，看两 run 是否拉开差距。
    2) λ 过大导致策略都是「几乎不用电池」：两 run 的 p_batt_applied_kw 是否都接近 0？
       排查：对比两 run 的 lambda_star、p_batt_applied_kw、efc_used；可试调大 EFC_EOL 或减小 safety_margin。
    3) 动作/网格过粗导致策略被量化成相同：排查：看单日 out_dp['P_applied_opt_kW'] 是否在两种输入下明显不同。
    """
    df = df_raw.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.set_index("datetime_utc").sort_index()

    system = build_system(cache_dir, Ns=Ns, Np=Np, N_series=N_series, N_parallel=N_parallel)

    # 固定 SOH + 预计算表（只做一次：极大加速滚动）
    # 使用与 SOC 相关的动作空间：soc_max 时只允许放电/零，soc_min 时只允许充电/零，避免边界选反方向
    cell = system.pack.cell
    soh_fixed = float(get_fixed_soh(cell))
    soc_grid = np.linspace(float(cell.soc_min), float(cell.soc_max), soc_grid_size).astype(np.float32)
    actions = build_soc_dependent_actions(system, soc_grid, action_grid_size=action_grid_size, strict_boundary=True)

    next_idx, p_applied, delta_efc = precompute_tables_parallel_cell_efc(
        system, soc_grid, actions, soh_fixed, dt_s=dt_s, n_jobs=n_jobs, backend="loky"
    )

    # 日滚动
    days = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq="D")
    all_days_out = []
    calib_log_rows = []

    lam_prev = None   # 用来存前一天的 λ*

    for day_idx, day_start in enumerate(days):
        # Set target EFC.
        # === 读取真实累计 EFC ===
        EFC_used_total = float(system.pack.cell.efc_cycle)
        EFC_remaining = max(EFC_EOL - EFC_used_total, 0.0)

        # === 剩余天数 ===
        days_remaining = len(days) - day_idx

        # === 今日硬预算 ===
        if days_remaining > 0:
            EFC_budget_day = (EFC_remaining / days_remaining) * (1.0 - safety_margin)
        else:
            EFC_budget_day = 0.0

        # -------------------------------------------------------------------------
        # 计算「下一天 action」时用到的数据（无校准时）：
        # (1) df_day_dp：当日 24h 的 Demand_Predicted, PV_Predicted, Price（仅预测，无当日真实负荷）
        # (2) system.soc：当日开始时 SOC，来自昨日仿真的 system.step 结果（依赖昨日真实 Demand/PV + 昨日 p_plan）
        # (3) system.pack.cell.efc_cycle：当前累计 EFC，来自历史仿真的 step（依赖历史真实 Demand/PV + 历史 p_plan）
        # (4) EFC_budget_day = (EFC_remaining / days_remaining) * (1 - safety_margin)，用于 bisect 的 target_efc
        # (5) lam_prev：昨日得到的 lam_star（昨日用昨日 df_day_dp 做 bisect 得到）
        # 因此：λ 由「满足 EFC 预算」唯一决定；若两 run 历史 EFC 用量相近，则 λ 相同；λ 很大时最优策略≈少用电池，与 net_load 几乎无关 → 两 run 表现相同。
        # -------------------------------------------------------------------------
        # 1) 当日 DP 输入：仅用 Demand_Predicted / PV_Predicted，无未来泄露
        df_day_dp = make_future_day_dp_df(df, day_start)

        # # For Test ONLY:
        # if day_start == pd.Timestamp('2021-12-27'):
        #     print(df_day_dp)
        #     # ================================================

        if debug_no_future_leak and day_idx == 0:
            day_slice = df.loc[day_start: day_start + pd.Timedelta(days=1) - pd.Timedelta(hours=1)]
            np.testing.assert_array_almost_equal(
                df_day_dp["Demand"].values, day_slice["Demand_Predicted"].values,
                err_msg="DP input Demand must come from Demand_Predicted (no leakage)."
            )
            np.testing.assert_array_almost_equal(
                df_day_dp["PV"].values, day_slice["PV_Predicted"].values,
                err_msg="DP input PV must come from PV_Predicted (no leakage)."
            )

        # === A) 求当日 λ（可选） ===
        # 默认：每日用 bisect 求一个“日预算对应的 λ”，然后校准日可能再覆盖成“校准后的 λ”
        # use_calibrated_lambda_only=True 时：禁用每日 bisect；仅在需要校准时更新 lam_prev；生成次日策略时直接用 lam_prev
        best_day = None
        if (not use_calibrated_lambda_only) or (lam_prev is None):
            if day_idx == 0 or lam_prev is None:
                best_day = bisect_lambda_for_efc_budget_vectorized(
                    df_day_dp, system, soc_grid, actions,
                    next_idx, p_applied, delta_efc,
                    target_efc=float(EFC_budget_day),
                    dt_s=dt_s,
                    terminal_equal_init=terminal_equal_init,
                    allow_export=allow_export,
                    max_iter=20,
                    tol=1e-4
                )
            else:
                best_day = bisect_lambda_warm_start(
                    df_day_dp, system, soc_grid, actions,
                    next_idx, p_applied, delta_efc,
                    target_efc=float(EFC_budget_day),
                    lam_init=float(lam_prev),
                    lam_expand=lam_expand_day,
                    dt_s=dt_s,
                    terminal_equal_init=terminal_equal_init,
                    allow_export=allow_export,
                    max_iter=max_iter_day,
                    tol=1e-4
                )

            lam_star = float(best_day["lam"])
            lam_calib = lam_star  # 非校准日沿用当日 λ
        else:
            # 已经有“上次校准 λ”：先占位，后面若 do_calib 会更新 lam_star/lam_calib
            lam_star = float(lam_prev)
            lam_calib = float(lam_prev)

        # ====== B) weekly calibration: 用过去几周 + 明天（长窗口）校准 λ ======
        # 只校准 lam_prev，让 λ 不被一天预测噪声带偏
        do_calib = (calib_every_days is not None) and (calib_every_days > 0) and (day_idx % calib_every_days == 0)

        if do_calib:
            df_win_long = make_search_window_df(df, day_start, past_weeks=past_weeks, check_output=check_search_window)
            window_hours = len(df_win_long) * (dt_s / 3600.0)

            # 关键：长窗口的 EFC 目标要按“每天预算 * 窗口长度/24”换算
            target_efc_long = float(EFC_budget_day) * (window_hours / 24.0)

            best_calib = bisect_lambda_warm_start(
                df_win_long, system, soc_grid, actions,
                next_idx, p_applied, delta_efc,
                target_efc=float(target_efc_long),
                lam_init=float(lam_star),          # 用今天的 λ 作为校准初值
                lam_expand=lam_expand_calib,
                dt_s=dt_s,
                terminal_equal_init=terminal_equal_init,
                allow_export=allow_export,
                max_iter=max_iter_calib,
                tol=1e-3
            )
            lam_star = float(best_calib["lam"])   # 用校准后的 λ 覆盖
            lam_calib = lam_star

        lam_prev = lam_star


        # 2) 固定 λ*，只对未来一天（预测 demand/pv）求策略
        # df_day_dp = make_future_day_dp_df(df, day_start)
        efc_day, cost_day, out_dp = solve_for_lambda_numba(
            df_day_dp, system, soc_grid, actions, next_idx, p_applied, delta_efc,
            dt_s=dt_s, terminal_equal_init=terminal_equal_init, allow_export=allow_export, lam=lam_star
        )
        p_plan = out_dp["P_applied_opt_kW"].to_numpy(dtype=float)

        # 3) 用真实电池模型 step 明天（真实 demand/pv）
        price_cols = ["Buy_Price", "Sell_Price"] if "Buy_Price" in df.columns and "Sell_Price" in df.columns else ["Price"]
        df_day_real = df.loc[day_start: day_start + pd.Timedelta(days=1) - pd.Timedelta(hours=1)][
            ["Demand", "PV"] + price_cols + ["Demand_Predicted", "PV_Predicted"]
        ].copy()
        if "Price" in df_day_real.columns and "Buy_Price" not in df_day_real.columns:
            df_day_real["Buy_Price"] = df_day_real["Price"]
            df_day_real["Sell_Price"] = df_day_real["Price"]
        out_sim = simulate_one_day_with_real_system(system, df_day_real, p_plan, dt_s=dt_s)
        efc_used_real_day = float(system.pack.cell.efc_cycle) - float(EFC_used_total)

        out_sim["lambda_star"] = lam_star
        out_sim["lambda_calib"] = float(lam_calib)
        out_sim["target_efc_window"] = float(EFC_budget_day)
        out_sim["efc_used"] = EFC_used_total
        # DP “日预算 bisect”阶段的 EFC（若禁用每日 bisect，则记为 NaN）
        out_sim["efc_used_window_dp"] = float(best_day["efc"]) if best_day is not None else float("nan")
        all_days_out.append(out_sim)

        if do_calib:
            calib_log_rows.append({
                "day_start": pd.to_datetime(day_start),
                "past_weeks": float(past_weeks),
                "window_hours": float(window_hours),
                "target_efc_day": float(EFC_budget_day),
                "target_efc_long": float(target_efc_long),
                "lambda_before_calib": float(best_day["lam"]) if best_day is not None else float("nan"),
                "lambda_after_calib": float(lam_calib),
                "dp_efc_long": float(best_calib.get("efc", float("nan"))),
                "efc_used_real_day": float(efc_used_real_day),
            })

    res = pd.concat(all_days_out, axis=0)
    res = res.reset_index().rename(columns={"index": "datetime_utc"})
    calib_log_df = pd.DataFrame(calib_log_rows).sort_values("day_start").reset_index(drop=True)
    if return_calibration_log:
        return res, calib_log_df
    return res
