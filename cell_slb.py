# Author: Kaifeng ZHU
# This file contains the main functions for battery cell model generation.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os, re
import hashlib
import pickle
import json
from pathlib import Path
from scipy.optimize import curve_fit, minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator

class LUT2D:
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df.index = df.index.astype(float)
        df.columns = df.columns.astype(float)
        df = df.sort_index()  # RegularGridInterpolator 要求递增
        self.soh_grid = df.index.values
        self.soc_grid = df.columns.values
        self.interp = RegularGridInterpolator(
            (self.soh_grid, self.soc_grid),
            df.values,
            bounds_error=False,
            fill_value=None,   # 允许外推；也可以改成 fill_value=...
        )
        self.soh_min, self.soh_max = self.soh_grid[0], self.soh_grid[-1]
        self.soc_min, self.soc_max = self.soc_grid[0], self.soc_grid[-1]

    def __call__(self, soh: float, soc: float) -> float:
        # 建议 clip，避免外推导致不合理
        soh_c = float(np.clip(soh, self.soh_min, self.soh_max))
        soc_c = float(np.clip(soc, self.soc_min, self.soc_max))
        return float(self.interp((soh_c, soc_c)))

# ---------- 2) 5min 内部仿真：给定恒功率 P，检查可行性 ----------
def feasible_constant_power(
    P_w: float,                     # 放电功率，W（放电取正；充电你可以取负或者单独写一个函数）
    soc0: float,
    soh: float,
    Qn_Ah: float,                   # 名义容量 Ah（单体）
    Vmin: float,
    Vmax: float,
    I_rated: float,                 # 单体电流上限 A
    dt_inner: float,                # 内部子步长 s（建议 1~10s）
    T: float,                       # 外部时长 300s
    lut_voc: LUT2D,
    lut_rint: LUT2D,
    lut_r1: LUT2D,
    lut_c1: LUT2D,
    lut_r2: LUT2D,
    lut_c2: LUT2D,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    mode: str = "discharge",
):

    # 初始 SoC 也要在范围内，否则无可行功率
    soc = float(np.clip(soc0, soc_min, soc_max))

    # 约定：放电 P_w >= 0
    if P_w < 0:
        return False

    Qa_Ah = soh * Qn_Ah  # 可用容量（Ah）
    if Qa_Ah <= 0:
        return False

    soc = float(np.clip(soc0, 0.0, 1.0))

    # 初始极化电压（可以设 0；更严谨可从上一步状态继承）
    V1 = 0.0
    V2 = 0.0

    # 初始电压用 Voc 近似
    Voc = lut_voc(soh, soc)
    V = Voc

    n_steps = int(np.ceil(T / dt_inner))
    dt = T / n_steps  # 精确覆盖 300s

    for _ in range(n_steps):
        # 参数插值
        Voc = lut_voc(soh, soc)
        Rint = max(lut_rint(soh, soc), 1e-6)
        R1   = max(lut_r1(soh, soc),   1e-8)
        C1   = max(lut_c1(soh, soc),   1e-8)
        R2   = max(lut_r2(soh, soc),   1e-8)
        C2   = max(lut_c2(soh, soc),   1e-8)

        tau1 = R1 * C1
        tau2 = R2 * C2

        # 恒功率 -> 电流（避免 V=0）
        V_safe = max(V, 0.1)
        if mode == "discharge":
            I = P_w / V_safe  # 放电电流（A）
        else:
            I = -P_w / V_safe  # 充电电流（A）

        # 电流上限
        if abs(I) > I_rated:
            return False

        # 2RC 精确离散（指数形式）
        a1 = np.exp(-dt / tau1) if tau1 > 1e-9 else 0.0
        a2 = np.exp(-dt / tau2) if tau2 > 1e-9 else 0.0

        V1 = a1 * V1 + (1 - a1) * R1 * I
        V2 = a2 * V2 + (1 - a2) * R2 * I

        # 端电压
        V = Voc - I * Rint - V1 - V2

        # 电压约束：放电主要检查 V>=Vmin，同时也避免超过 Vmax（异常）
        if (V < Vmin) or (V > Vmax):
            return False

        # SoC 更新（库仑计数）
        # 放电：SoC 下降
        soc = soc - (I * dt) / (Qa_Ah * 3600.0)
        # 改这里：SoC 运行区间限制
        if (soc < soc_min) or (soc > soc_max):
            return False

    return True

# ---------- 3) 二分搜索得到 Pmax(5min) ----------
def compute_pmax(
    soc0: float,
    soh: float,
    P_upper_w: float,               # 搜索上界（W），比如基于 I_rated*Voc 的粗上界
    Qn_Ah: float,
    Vmin: float,
    Vmax: float,
    I_rated: float,
    dt_inner: float,
    T: float,
    luts: dict,
    tol_w: float = 5.0,             # 功率精度（W）
    max_iter: int = 40,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    mode: str = "discharge",
):
    lo, hi = 0.0, float(P_upper_w)

    # 如果上界都可行，直接返回上界
    if feasible_constant_power(
        hi, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated, dt_inner, T,
        luts["Voc"], luts["Rint"], luts["R1"], luts["C1"], luts["R2"], luts["C2"],
        soc_min=soc_min, soc_max=soc_max, mode=mode
    ):
        return hi

    # 二分
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ok = feasible_constant_power(
            mid, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated, dt_inner, T,
            luts["Voc"], luts["Rint"], luts["R1"], luts["C1"], luts["R2"], luts["C2"],
            soc_min=soc_min, soc_max=soc_max, mode=mode
        )
        if ok:
            lo = mid
        else:
            hi = mid

        if (hi - lo) < tol_w:
            break

    return lo

# ---------- 2.5) 新方法：通过积分取平均计算最大功率 ----------
def feasible_variable_power(
    power_profile: np.ndarray,  # 功率曲线 [P1, P2, ..., Pn]
    soc0: float,
    soh: float,
    Qn_Ah: float,
    Vmin: float,
    Vmax: float,
    I_rated: float,
    dt_inner: float,
    T: float,
    lut_voc: LUT2D,
    lut_rint: LUT2D,
    lut_r1: LUT2D,
    lut_c1: LUT2D,
    lut_r2: LUT2D,
    lut_c2: LUT2D,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    mode: str = "discharge",
):
    """
    检查变化的功率曲线是否可行。
    功率曲线在时间窗口内可以变化，只要满足所有约束即可。
    
    Args:
        power_profile: 功率曲线数组，长度为 n_steps
        其他参数同 feasible_constant_power
    
    Returns:
        (is_feasible, average_power, details_dict)
    """
    soc = float(np.clip(soc0, soc_min, soc_max))
    Qa_Ah = soh * Qn_Ah
    if Qa_Ah <= 0:
        return False, 0.0, {}
    
    # 初始化
    V1 = 0.0
    V2 = 0.0
    Voc = lut_voc(soh, soc)
    V = Voc
    
    n_segments = len(power_profile)
    # 每个段的时长
    dt_segment = T / n_segments
    # 每个段内的内部步数
    n_inner_per_segment = max(1, int(np.ceil(dt_segment / dt_inner)))
    dt = dt_segment / n_inner_per_segment
    
    # 记录信息
    voltages = []
    currents = []
    socs = []
    powers = []
    
    # 对每个功率段进行仿真
    for seg_idx, P_w in enumerate(power_profile):
        # 符号约定：放电功率为正，充电功率也为正（在build_pmax_lut中已处理）
        # 这里只需要确保功率非负
        if mode == "discharge" and P_w < 0:
            P_w = 0.0
        elif mode == "charge" and P_w < 0:
            P_w = 0.0
        
        # 在这个段内，进行多个内部步的仿真
        for inner_step in range(n_inner_per_segment):
            # 插值参数
            Voc = lut_voc(soh, soc)
            Rint = max(lut_rint(soh, soc), 1e-6)
            R1 = max(lut_r1(soh, soc), 1e-8)
            C1 = max(lut_c1(soh, soc), 1e-8)
            R2 = max(lut_r2(soh, soc), 1e-8)
            C2 = max(lut_c2(soh, soc), 1e-8)
            
            tau1 = R1 * C1
            tau2 = R2 * C2
            
            # 计算电流
            V_safe = max(V, 0.1)
            if mode == "discharge":
                I = P_w / V_safe if P_w > 0.01 else 0.0
            else:
                I = -P_w / V_safe if P_w > 0.01 else 0.0
            
            # 检查电流约束
            if abs(I) > I_rated:
                return False, 0.0, {}
            
            # 更新RC电路
            a1 = np.exp(-dt / tau1) if tau1 > 1e-9 else 0.0
            a2 = np.exp(-dt / tau2) if tau2 > 1e-9 else 0.0
            V1 = a1 * V1 + (1 - a1) * R1 * I
            V2 = a2 * V2 + (1 - a2) * R2 * I
            
            # 计算端电压
            V = Voc - I * Rint - V1 - V2
            
            # 检查电压约束
            if (V < Vmin) or (V > Vmax):
                return False, 0.0, {}
            
            # 更新SOC
            soc = soc - (I * dt) / (Qa_Ah * 3600.0)
            if (soc < soc_min) or (soc > soc_max):
                return False, 0.0, {}
            
            # 记录（只在每个段的最后一步记录，或者每步都记录）
            if inner_step == n_inner_per_segment - 1:  # 段的最后一步
                voltages.append(V)
                currents.append(I)
                socs.append(soc)
                powers.append(P_w)
    
    # 计算平均功率（积分取平均）
    # P_avg = (1/T) * ∫P(t)dt ≈ (1/T) * Σ(P_i * dt_segment_i)
    # 每个段的时长为 dt_segment，功率为 P_i
    average_power = np.sum(power_profile) * dt_segment / T
    
    details = {
        'voltages': np.array(voltages),
        'currents': np.array(currents),
        'socs': np.array(socs),
        'powers': np.array(powers),
    }
    
    return True, average_power, details


def compute_pmax_average(
    soc0: float,
    soh: float,
    P_upper_w: float,
    Qn_Ah: float,
    Vmin: float,
    Vmax: float,
    I_rated: float,
    dt_inner: float,
    T: float,
    luts: dict,
    n_segments: int = 5,  # 将时间窗口分成n_segments段（减少到5以提高速度）
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    mode: str = "discharge",
    max_opt_iter: int = 30,  # 减少优化迭代次数
    opt_tol: float = 0.01,  # 放宽优化容差（相对值，如1%）
    initial_power_profile: np.ndarray = None,  # 使用之前的结果作为初始值
    cache: dict = None,  # 缓存字典，用于存储已计算的结果
):
    """
    通过优化功率曲线来计算最大平均功率。
    使用积分取平均：P_avg = (1/T) * ∫P(t)dt
    
    Args:
        n_segments: 将时间窗口T分成多少段，每段内功率可以不同
        max_opt_iter: 最大优化迭代次数（减少以提高速度）
        opt_tol: 优化容差（相对值，如0.01表示1%）
        initial_power_profile: 初始功率曲线（如果提供，用于加速优化）
        cache: 缓存字典，格式：{(soh, soc): (power, profile)}
        其他参数同 compute_pmax
    
    Returns:
        最大可行的平均功率
    """
    from scipy.optimize import minimize, Bounds
    
    # 检查缓存
    cache_key = (soh, soc0)
    if cache is not None and cache_key in cache:
        cached_power, cached_profile = cache[cache_key]
        # 如果缓存的结果在合理范围内，直接返回
        if cached_power > 0:
            return cached_power
    
    # 完全独立于恒功率方法，使用优化算法直接找到最大平均功率
    # 目标函数：最大化平均功率（即最小化负的平均功率）
    best_power = 0.0  # 从0开始，不依赖恒功率结果
    best_profile = None
    n_eval = 0
    
    def objective(power_array):
        nonlocal best_power, best_profile, n_eval
        n_eval += 1
        
        # power_array: [P1, P2, ..., Pn_segments]
        is_feasible, avg_power, _ = feasible_variable_power(
            power_array, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated,
            dt_inner, T, luts["Voc"], luts["Rint"], luts["R1"],
            luts["C1"], luts["R2"], luts["C2"],
            soc_min=soc_min, soc_max=soc_max, mode=mode
        )
        
        # 如果不可行，返回一个很大的惩罚值
        if not is_feasible:
            return 1e6
        
        # 更新最佳结果
        if avg_power > best_power:
            best_power = avg_power
            best_profile = power_array.copy()
        
        # 返回负的平均功率（因为要最小化）
        return -avg_power
    
    # 初始猜测：优先使用提供的初始值，否则使用基于I_rated和Voc的估计
    if initial_power_profile is not None and len(initial_power_profile) == n_segments:
        # 使用之前的结果作为初始值（可能来自相邻SOC点）
        initial_power = initial_power_profile.copy()
        # 确保在合理范围内
        initial_power = np.clip(initial_power, 0.0, P_upper_w)
    else:
        # 使用基于I_rated和Voc的估计作为初始值（不依赖恒功率方法）
        Voc0 = luts["Voc"](soh, soc0)
        # 估计初始功率：基于I_rated和Voc的粗略估计
        P_estimate = I_rated * Voc0 * 0.8  # 使用80%作为保守估计
        P_estimate = min(P_estimate, P_upper_w)
        initial_power = np.full(n_segments, P_estimate)
    
    # 约束：功率在合理范围内
    bounds = Bounds(lb=0.0, ub=P_upper_w)
    
    # 优化（使用更快的设置）
    try:
        # 计算绝对容差（基于P_upper的百分比）
        abs_tol = P_upper_w * opt_tol if P_upper_w > 0 else 0.01
        
        # 如果初始值导致不可行，尝试从更小的功率开始
        # 先检查初始值是否可行
        is_feasible_init, _, _ = feasible_variable_power(
            initial_power, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated,
            dt_inner, T, luts["Voc"], luts["Rint"], luts["R1"],
            luts["C1"], luts["R2"], luts["C2"],
            soc_min=soc_min, soc_max=soc_max, mode=mode
        )
        
        # 如果初始值不可行，使用二分搜索找到可行范围
        if not is_feasible_init:
            # 二分搜索找到最大可行功率（作为初始值）
            lo, hi = 0.0, P_upper_w
            for _ in range(20):  # 最多20次二分
                mid = 0.5 * (lo + hi)
                test_power = np.full(n_segments, mid)
                is_feasible, _, _ = feasible_variable_power(
                    test_power, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated,
                    dt_inner, T, luts["Voc"], luts["Rint"], luts["R1"],
                    luts["C1"], luts["R2"], luts["C2"],
                    soc_min=soc_min, soc_max=soc_max, mode=mode
                )
                if is_feasible:
                    lo = mid
                    initial_power = np.full(n_segments, mid)
                else:
                    hi = mid
                if (hi - lo) < 0.01:  # 精度足够
                    break
        
        # 尝试多个初始值策略，确保能找到可行解
        best_result_power = 0.0
        best_result_profile = None
        
        # 策略1：使用提供的初始值或估计值
        initial_values_to_try = [initial_power]
        
        # 策略2：如果初始值不可行，尝试从更小的功率开始
        if not is_feasible_init:
            # 尝试多个不同的初始功率值
            for scale in [0.1, 0.2, 0.3, 0.5, 0.7]:
                test_init = np.full(n_segments, P_upper_w * scale)
                is_feasible_test, _, _ = feasible_variable_power(
                    test_init, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated,
                    dt_inner, T, luts["Voc"], luts["Rint"], luts["R1"],
                    luts["C1"], luts["R2"], luts["C2"],
                    soc_min=soc_min, soc_max=soc_max, mode=mode
                )
                if is_feasible_test:
                    initial_values_to_try.append(test_init)
                    break  # 找到一个可行的就停止
        
        # 对每个初始值尝试优化
        for init_power in initial_values_to_try:
            # 重置best_power用于这次尝试
            current_best = 0.0
            current_profile = None
            
            def objective_local(power_array):
                nonlocal current_best, current_profile
                is_feasible, avg_power, _ = feasible_variable_power(
                    power_array, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated,
                    dt_inner, T, luts["Voc"], luts["Rint"], luts["R1"],
                    luts["C1"], luts["R2"], luts["C2"],
                    soc_min=soc_min, soc_max=soc_max, mode=mode
                )
                if not is_feasible:
                    return 1e6
                if avg_power > current_best:
                    current_best = avg_power
                    current_profile = power_array.copy()
                return -avg_power
            
            try:
                result = minimize(
                    objective_local,
                    init_power,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_opt_iter,
                        'ftol': abs_tol,
                        'gtol': abs_tol,
                    }
                )
                
                # 更新全局最佳结果
                if current_best > best_result_power:
                    best_result_power = current_best
                    best_result_profile = current_profile
                    best_power = current_best
                    best_profile = current_profile
            except:
                continue
        
        # 完全依赖优化结果
        final_power = best_result_power
        final_profile = best_result_profile
        
        # 如果优化完全失败，使用更robust的二分搜索找到最大可行功率
        if final_power <= 0:
            # 首先检查是否有任何可行的功率（即使很小）
            # 从非常小的功率开始测试
            test_powers = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
            best_feasible = 0.0
            best_feasible_profile = None
            
            for test_p in test_powers:
                if test_p > P_upper_w:
                    break
                # 尝试不同的功率分布策略
                test_profiles = [
                    np.full(n_segments, test_p),  # 恒功率
                ]
                # 如果功率较大，尝试变化分布
                if test_p > 0.1:
                    test_profiles.extend([
                        np.linspace(test_p * 1.1, test_p * 0.9, n_segments),  # 递减
                        np.linspace(test_p * 0.9, test_p * 1.1, n_segments),  # 递增
                    ])
                
                for test_profile in test_profiles:
                    test_profile = np.clip(test_profile, 0.0, P_upper_w)
                    is_feasible, avg_power, _ = feasible_variable_power(
                        test_profile, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated,
                        dt_inner, T, luts["Voc"], luts["Rint"], luts["R1"],
                        luts["C1"], luts["R2"], luts["C2"],
                        soc_min=soc_min, soc_max=soc_max, mode=mode
                    )
                    if is_feasible and avg_power > best_feasible:
                        best_feasible = avg_power
                        best_feasible_profile = test_profile.copy()
            
            # 如果找到了可行解，使用二分搜索进一步优化
            if best_feasible > 0:
                # 在找到的可行功率基础上，使用二分搜索找到最大可行功率
                lo = best_feasible
                hi = min(P_upper_w, best_feasible * 2)  # 从找到的功率开始向上搜索
                
                for _ in range(20):  # 最多20次二分
                    mid = 0.5 * (lo + hi)
                    test_profiles = [
                        np.full(n_segments, mid),
                        np.linspace(mid * 1.1, mid * 0.9, n_segments),
                        np.linspace(mid * 0.9, mid * 1.1, n_segments),
                    ]
                    
                    found_higher = False
                    for test_profile in test_profiles:
                        test_profile = np.clip(test_profile, 0.0, P_upper_w)
                        is_feasible, avg_power, _ = feasible_variable_power(
                            test_profile, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated,
                            dt_inner, T, luts["Voc"], luts["Rint"], luts["R1"],
                            luts["C1"], luts["R2"], luts["C2"],
                            soc_min=soc_min, soc_max=soc_max, mode=mode
                        )
                        if is_feasible and avg_power > best_feasible:
                            best_feasible = avg_power
                            best_feasible_profile = test_profile.copy()
                            found_higher = True
                    
                    if found_higher:
                        lo = mid
                    else:
                        hi = mid
                    
                    if (hi - lo) < 0.01:
                        break
                
                final_power = best_feasible
                final_profile = best_feasible_profile
        
        # 缓存结果
        if cache is not None:
            cache[cache_key] = (final_power, final_profile)
        
        return final_power
    except:
        # 如果出错，尝试使用一个很小的功率值
        # 检查非常小的功率是否可行
        try:
            small_power = np.full(n_segments, 0.1)  # 0.1W
            is_feasible_small, avg_power_small, _ = feasible_variable_power(
                small_power, soc0, soh, Qn_Ah, Vmin, Vmax, I_rated,
                dt_inner, T, luts["Voc"], luts["Rint"], luts["R1"],
                luts["C1"], luts["R2"], luts["C2"],
                soc_min=soc_min, soc_max=soc_max, mode=mode
            )
            if is_feasible_small and avg_power_small > 0:
                if cache is not None:
                    cache[cache_key] = (avg_power_small, small_power)
                return avg_power_small
        except:
            pass
        
        # 如果都失败，返回0
        if cache is not None:
            cache[cache_key] = (0.0, None)
        return 0.0


# ---------- 4) 生成 LUT（方案C核心） ----------
def _compute_single_pmax(
    soh: float,
    soc: float,
    Voc_df, Rint_df, R1_df, C1_df, R2_df, C2_df,
    Qn_Ah: float,
    Vmin: float,
    Vmax: float,
    I_rated: float,
    dt_inner: float,
    T: float,
    P_upper_factor: float,
    soc_min: float,
    soc_max: float,
    mode: str,
    use_average_power: bool,
    n_segments: int,
    max_opt_iter: int,
    opt_tol: float,
):
    """Worker function for parallel computation of a single (soh, soc) point."""
    # 在worker中重建LUT2D对象（因为LUT2D可能不能直接pickle）
    luts = {
        "Voc": LUT2D(Voc_df),
        "Rint": LUT2D(Rint_df),
        "R1": LUT2D(R1_df),
        "C1": LUT2D(C1_df),
        "R2": LUT2D(R2_df),
        "C2": LUT2D(C2_df),
    }
    
    Voc0 = luts["Voc"](soh, soc)
    P_upper = P_upper_factor * I_rated * max(Voc0, Vmin)
    
    # 检查SOC范围
    if (soc < soc_min) or (soc > soc_max):
        return 0.0
    
    # 选择计算方法
    if use_average_power:
        # 使用积分取平均的方法（并行时不能使用prev_profile，但可以使用缓存）
        p = compute_pmax_average(
            soc0=soc, soh=soh,
            P_upper_w=P_upper,
            Qn_Ah=Qn_Ah, Vmin=Vmin, Vmax=Vmax,
            I_rated=I_rated,
            dt_inner=dt_inner, T=T,
            luts=luts,
            n_segments=n_segments,
            soc_min=soc_min, soc_max=soc_max,
            mode=mode,
            max_opt_iter=max_opt_iter,
            opt_tol=opt_tol,
            initial_power_profile=None,  # 并行时不能使用prev_profile
            cache=None,  # 并行时每个worker独立，不使用共享缓存
        )
    else:
        # 使用恒功率方法（原方法）
        p = compute_pmax(
            soc0=soc, soh=soh,
            P_upper_w=P_upper,
            Qn_Ah=Qn_Ah, Vmin=Vmin, Vmax=Vmax,
            I_rated=I_rated,
            dt_inner=dt_inner, T=T,
            luts=luts,
            mode=mode
        )
    
    return p


def build_pmax_lut(
    Voc_df, Rint_df, R1_df, C1_df, R2_df, C2_df,
    Qn_Ah: float,
    Vmin: float,
    Vmax: float,
    I_rated: float,
    dt_inner: float = 5.0,
    T: float = 300.0,
    P_upper_factor: float = 1.2,    # 上界倍率，避免太小
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    mode: str = "discharge",
    use_average_power: bool = False,  # 是否使用积分取平均的方法
    n_segments: int = 5,  # 如果使用平均功率方法，时间窗口分段数（减少以提高速度）
    max_opt_iter: int = 30,  # 优化最大迭代次数
    opt_tol: float = 0.01,  # 优化容差（相对值）
    use_parallel: bool = False,  # 是否使用并行计算
    n_jobs: int = -1,  # 并行任务数，-1表示使用所有CPU核心
):
    # 构造插值器
    luts = {
        "Voc": LUT2D(Voc_df),
        "Rint": LUT2D(Rint_df),
        "R1": LUT2D(R1_df),
        "C1": LUT2D(C1_df),
        "R2": LUT2D(R2_df),
        "C2": LUT2D(C2_df),
    }

    soh_grid = np.array(sorted(Voc_df.index.astype(float)))
    soc_grid = np.array(sorted(Voc_df.columns.astype(float)))

    Pmax = pd.DataFrame(index=soh_grid, columns=soc_grid, dtype=float)

    if use_parallel:
        # 并行计算
        try:
            from joblib import Parallel, delayed
            
            # 生成所有(soh, soc)组合
            tasks = []
            for soh in soh_grid:
                for soc in soc_grid:
                    tasks.append((soh, soc))
            
            # 并行计算
            results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(_compute_single_pmax)(
                    soh, soc,
                    Voc_df, Rint_df, R1_df, C1_df, R2_df, C2_df,
                    Qn_Ah, Vmin, Vmax, I_rated,
                    dt_inner, T, P_upper_factor,
                    soc_min, soc_max, mode,
                    use_average_power, n_segments,
                    max_opt_iter, opt_tol,
                )
                for soh, soc in tasks
            )
            
            # 填充结果到DataFrame
            idx = 0
            for soh in soh_grid:
                for soc in soc_grid:
                    Pmax.loc[soh, soc] = results[idx]
                    idx += 1
                    
        except ImportError:
            print("Warning: joblib not available, falling back to serial computation.")
            print("Install joblib with: pip install joblib")
            use_parallel = False
        except Exception as e:
            print(f"Warning: Parallel computation failed: {e}")
            print("Falling back to serial computation.")
            use_parallel = False
    
    if not use_parallel:
        # 串行计算（优化：从低SOC开始，使用缓存和之前的结果作为初始值）
        # 构造插值器
        luts = {
            "Voc": LUT2D(Voc_df),
            "Rint": LUT2D(Rint_df),
            "R1": LUT2D(R1_df),
            "C1": LUT2D(C1_df),
            "R2": LUT2D(R2_df),
            "C2": LUT2D(C2_df),
        }
        
        # 创建缓存字典：{(soh, soc): (power, profile)}
        cache = {} if use_average_power else None
        
        for soh in soh_grid:
            # 对每个SOH，从低SOC开始计算（利用相邻SOC点的相似性）
            prev_profile = None  # 存储前一个SOC点的功率曲线
            
            for soc in soc_grid:
                Voc0 = luts["Voc"](soh, soc)
                # 粗上界：由电流上限给一个功率上界（放电用 Vmin 或 Voc 都行）
                P_upper = P_upper_factor * I_rated * max(Voc0, Vmin)

                # 选择计算方法
                if use_average_power:
                    # 使用积分取平均的方法，传入之前的结果作为初始值
                    p = compute_pmax_average(
                        soc0=soc, soh=soh,
                        P_upper_w=P_upper,
                        Qn_Ah=Qn_Ah, Vmin=Vmin, Vmax=Vmax,
                        I_rated=I_rated,
                        dt_inner=dt_inner, T=T,
                        luts=luts,
                        n_segments=n_segments,
                        soc_min=soc_min, soc_max=soc_max,
                        mode=mode,
                        max_opt_iter=max_opt_iter,
                        opt_tol=opt_tol,
                        initial_power_profile=prev_profile,  # 使用前一个SOC点的结果
                        cache=cache,  # 传入缓存
                    )
                    
                    # 更新prev_profile用于下一个SOC点（从缓存中获取）
                    cache_key = (soh, soc)
                    if cache_key in cache:
                        _, cached_profile = cache[cache_key]
                        if cached_profile is not None:
                            prev_profile = cached_profile
                else:
                    # 使用恒功率方法（原方法）
                    p = compute_pmax(
                        soc0=soc, soh=soh,
                        P_upper_w=P_upper,
                        Qn_Ah=Qn_Ah, Vmin=Vmin, Vmax=Vmax,
                        I_rated=I_rated,
                        dt_inner=dt_inner, T=T,
                        luts=luts,
                        mode=mode
                    )

                # 改这里：SoC 运行区间限制
                if (soc < soc_min) or (soc > soc_max):
                    p = 0.0

                Pmax.loc[soh, soc] = p

    return Pmax

def current_from_power_discharge(P_out_w: float, Voc: float, R: float) -> float:
    """放电：P_out = I*(Voc - I*R), I>=0，返回电流幅值"""
    P_out_w = max(0.0, float(P_out_w))
    R = max(float(R), 1e-9)
    disc = Voc*Voc - 4.0*R*P_out_w
    disc = max(disc, 0.0)
    return (Voc - disc**0.5) / (2.0*R)

def current_from_power_charge(P_in_w: float, Voc: float, R: float) -> float:
    """充电：P_in = I*(Voc + I*R), I>=0，返回电流幅值"""
    P_in_w = max(0.0, float(P_in_w))
    R = max(float(R), 1e-9)
    disc = Voc*Voc + 4.0*R*P_in_w
    return (-Voc + disc**0.5) / (2.0*R)


@dataclass
class CellSLB:
    """
    Physics-based cell model:
    - Energy-based SOC (from integrated measured V/I)
    - Step power limits derived from data (max avg power over dt window vs SOC)
    Sign convention:
      - raw current i [A] from data: positive = charge, negative = discharge (as in many cyclers)
      - power p = v * i [W]: positive = charging into cell, negative = discharging out of cell
    """
    # df: pd.DataFrame                # Use the dataset in initialization state!!
    # data_folder: str
    file_path: str
    dt_step_s: float = 300.0         # simulation/control step, e.g. 5 min
    soc_init: float = 0.5            # initial SOC in [0,1]
    # soc_bins: int = 101              # number of SOC grid edges (bins-1 table points)
    soc_min: float = 0.0
    soc_max: float = 1.0
    degradation_column_name: str = 'SOH_1'
    nominal_capacity_Ah: float = 2.1
    max_charge_current: float = 2.1 # Unit in A
    max_discharge_current: float = -10  # Unit in A
    dt_inner_power_limit: float = 5.0
    use_average_power_method: bool = False  # 是否使用积分取平均的方法计算Pmax
    n_power_segments: int = 5  # 如果使用平均功率方法，时间窗口分段数（减少以提高速度）
    max_opt_iter: int = 30  # 优化最大迭代次数
    opt_tol: float = 0.01  # 优化容差（相对值，如0.01表示1%）
    use_parallel_computation: bool = False  # 是否使用并行计算
    n_jobs: int = -1  # 并行任务数，-1表示使用所有CPU核心，1表示串行
    cache_dir: str = None  # 缓存目录，如果为None则不使用缓存
    use_cache: bool = True  # 是否使用缓存（如果cache_dir为None则自动禁用）
    # 在构建用于功率极限计算的 LUT 之前，先把原始 Excel LUT
    # 在 SOC 轴上重采样到一个新的网格。
    # 这里可以通过 n_soc_grid_points 来控制 [soc_min, soc_max] 区间内的点数。
    # 如果 <= 0，则不做重采样，直接使用原始 Excel 的 SOC 网格。
    n_soc_grid_points: int = 0


    # computed fields
    r_init_charge_df: pd.DataFrame = None
    r_init_discharge_df: pd.DataFrame = None
    r1_charge_df: pd.DataFrame = None
    r1_discharge_df: pd.DataFrame = None
    r2_charge_df: pd.DataFrame = None
    r2_discharge_df: pd.DataFrame = None
    c1_charge_df: pd.DataFrame = None
    c1_discharge_df: pd.DataFrame = None
    c2_charge_df: pd.DataFrame = None
    c2_discharge_df: pd.DataFrame = None
    v_oc_df: pd.DataFrame = None
    available_degradation_df: pd.DataFrame = None
    degradation_df: pd.DataFrame = None
    soh: float = None
    soc: float = None
    capacity_Ah: float = None
    efc_cycle: float = None
    luts: dict = None



    def __post_init__(self):
        self._prepare_dataframe_init()

    # ---------------------------
    # Cache management
    # ---------------------------
    def _get_cache_key(self) -> str:
        """生成缓存key（基于所有相关参数）"""
        # 获取文件修改时间作为数据源标识
        file_mtime = os.path.getmtime(self.file_path) if os.path.exists(self.file_path) else 0
        
        # 构建参数字符串
        params_str = (
            f"{self.file_path}_{file_mtime}_"
            f"{self.dt_step_s}_{self.soc_min}_{self.soc_max}_"
            f"{self.nominal_capacity_Ah}_{self.max_charge_current}_{self.max_discharge_current}_"
            f"{self.dt_inner_power_limit}_{self.use_average_power_method}_"
            f"{self.n_power_segments}_{self.max_opt_iter}_{self.opt_tol}_"
            f"{self.use_parallel_computation}_{self.n_jobs}_{self.degradation_column_name}"
        )
        
        # 生成hash
        hash_obj = hashlib.md5(params_str.encode())
        return hash_obj.hexdigest()
    
    def _get_cache_path(self, mode: str) -> Path:
        """获取缓存文件路径"""
        if not self.use_cache or self.cache_dir is None:
            return None
        
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_key = self._get_cache_key()
        return cache_dir / f"power_limit_{mode}_{cache_key}.pkl"
    
    def _get_params_json_path(self) -> Path:
        """获取参数JSON文件路径"""
        if not self.use_cache or self.cache_dir is None:
            return None
        
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_key = self._get_cache_key()
        return cache_dir / f"cell_params_{cache_key}.json"
    
    def _get_all_params(self) -> dict:
        """获取所有相关参数（用于保存和验证）"""
        # 获取文件修改时间
        file_mtime = os.path.getmtime(self.file_path) if os.path.exists(self.file_path) else 0
        
        return {
            "file_path": str(self.file_path),
            "file_mtime": file_mtime,
            "dt_step_s": self.dt_step_s,
            "soc_init": self.soc_init,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "degradation_column_name": self.degradation_column_name,
            "nominal_capacity_Ah": self.nominal_capacity_Ah,
            "max_charge_current": self.max_charge_current,
            "max_discharge_current": self.max_discharge_current,
            "dt_inner_power_limit": self.dt_inner_power_limit,
            "use_average_power_method": self.use_average_power_method,
            "n_power_segments": self.n_power_segments,
            "max_opt_iter": self.max_opt_iter,
            "opt_tol": self.opt_tol,
            "use_parallel_computation": self.use_parallel_computation,
            "n_jobs": self.n_jobs,
        }
    
    def _verify_params(self, cached_params: dict) -> tuple[bool, list]:
        """
        验证当前参数与缓存参数是否一致
        
        返回:
            (is_match, differences) - is_match表示是否匹配，differences是差异列表
        """
        current_params = self._get_all_params()
        differences = []
        
        # 检查每个参数
        for key in current_params:
            if key not in cached_params:
                differences.append(f"Missing parameter in cache: {key}")
            elif current_params[key] != cached_params[key]:
                # 对于浮点数，使用相对容差比较
                if isinstance(current_params[key], float) and isinstance(cached_params[key], float):
                    if abs(current_params[key] - cached_params[key]) > 1e-10:
                        differences.append(
                            f"{key}: current={current_params[key]}, cached={cached_params[key]}"
                        )
                else:
                    differences.append(
                        f"{key}: current={current_params[key]}, cached={cached_params[key]}"
                    )
        
        # 检查是否有缓存中有但当前没有的参数
        for key in cached_params:
            if key not in current_params:
                differences.append(f"Extra parameter in cache: {key}")
        
        return len(differences) == 0, differences
    
    def save_power_limits(self, cache_dir: str = None):
        """
        保存power limit到缓存文件，同时保存参数JSON文件
        
        参数:
            cache_dir: 缓存目录，如果为None则使用self.cache_dir
        """
        if cache_dir is not None:
            self.cache_dir = cache_dir
        
        if not self.use_cache or self.cache_dir is None:
            print("Warning: cache_dir is None, cannot save power limits.")
            return
        
        # 保存参数JSON文件
        params_path = self._get_params_json_path()
        if params_path:
            params = self._get_all_params()
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            # print(f"Saved cell parameters to: {params_path}")
        
        # 保存放电power limit
        discharge_path = self._get_cache_path("discharge")
        if discharge_path:
            with open(discharge_path, 'wb') as f:
                pickle.dump(self.p_max_discharge_df, f)
            # print(f"Saved discharge power limit to: {discharge_path}")
        
        # 保存充电power limit
        charge_path = self._get_cache_path("charge")
        if charge_path:
            with open(charge_path, 'wb') as f:
                pickle.dump(self.p_max_charge_df, f)
            # print(f"Saved charge power limit to: {charge_path}")
    
    def _load_power_limits(self) -> tuple:
        """
        从缓存文件加载power limit，先验证参数是否一致
        
        返回:
            (discharge_df, charge_df) 如果加载成功，否则返回 (None, None)
        """
        if not self.use_cache or self.cache_dir is None:
            return None, None
        
        # 先检查参数JSON文件是否存在
        params_path = self._get_params_json_path()
        if params_path and params_path.exists():
            try:
                with open(params_path, 'r', encoding='utf-8') as f:
                    cached_params = json.load(f)
                
                # 验证参数是否一致
                is_match, differences = self._verify_params(cached_params)
                if not is_match:
                    print("=" * 70)
                    print("Warning: Cached parameters do not match current parameters!")
                    print("The following parameters differ:")
                    for diff in differences:
                        print(f"  - {diff}")
                    print("=" * 70)
                    print("Will recompute power limits instead of using cache.")
                    return None, None
                # else:
                #     print("✓ Parameters match cached values. Loading from cache...")
            except Exception as e:
                print(f"Warning: Failed to load/verify parameters JSON: {e}")
                print("Will recompute power limits instead of using cache.")
                return None, None
        else:
            # 如果没有参数JSON文件，说明缓存可能不完整，不加载
            print("No parameters JSON file found. Will recompute power limits.")
            return None, None
        
        discharge_path = self._get_cache_path("discharge")
        charge_path = self._get_cache_path("charge")
        
        discharge_df = None
        charge_df = None
        
        # 尝试加载放电power limit
        if discharge_path and discharge_path.exists():
            try:
                with open(discharge_path, 'rb') as f:
                    discharge_df = pickle.load(f)
                # print(f"✓ Loaded discharge power limit from: {discharge_path}")
            except Exception as e:
                print(f"Warning: Failed to load discharge power limit: {e}")
        
        # 尝试加载充电power limit
        if charge_path and charge_path.exists():
            try:
                with open(charge_path, 'rb') as f:
                    charge_df = pickle.load(f)
                # print(f"✓ Loaded charge power limit from: {charge_path}")
            except Exception as e:
                print(f"Warning: Failed to load charge power limit: {e}")
        
        return discharge_df, charge_df
    
    @classmethod
    def from_json(cls, json_path: str, cache_dir: str = None, use_cache: bool = True):
        """
        从JSON文件创建CellSLB对象
        
        参数:
            json_path: JSON文件路径（例如："./cell_cache/cell_params_xxx.json"）
            cache_dir: 缓存目录，如果为None则从JSON文件所在目录推断
            use_cache: 是否使用缓存
        
        返回:
            CellSLB对象
        
        示例:
            cell = CellSLB.from_json("./cell_cache/cell_params_xxx.json")
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # 如果cache_dir未指定，使用JSON文件所在目录
        if cache_dir is None:
            cache_dir = str(json_path.parent)
        
        # 从JSON中提取参数（排除file_mtime，因为它是运行时计算的）
        cell_params = {
            "file_path": params["file_path"],
            "dt_step_s": params["dt_step_s"],
            "soc_init": params["soc_init"],
            "soc_min": params["soc_min"],
            "soc_max": params["soc_max"],
            "degradation_column_name": params["degradation_column_name"],
            "nominal_capacity_Ah": params["nominal_capacity_Ah"],
            "max_charge_current": params["max_charge_current"],
            "max_discharge_current": params["max_discharge_current"],
            "dt_inner_power_limit": params["dt_inner_power_limit"],
            "use_average_power_method": params["use_average_power_method"],
            "n_power_segments": params["n_power_segments"],
            "max_opt_iter": params["max_opt_iter"],
            "opt_tol": params["opt_tol"],
            "use_parallel_computation": params["use_parallel_computation"],
            "n_jobs": params["n_jobs"],
            "cache_dir": cache_dir,
            "use_cache": use_cache,
        }
        
        # 创建CellSLB对象
        cell = cls(**cell_params)
        
        # print(f"✓ Created CellSLB from JSON: {json_path}")
        # print(f"  Cache directory: {cache_dir}")
        
        return cell
    
    @staticmethod
    def find_json_files(cache_dir: str) -> list:
        """
        在缓存目录中查找所有JSON参数文件
        
        参数:
            cache_dir: 缓存目录路径
        
        返回:
            JSON文件路径列表
        """
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            return []
        
        json_files = list(cache_dir.glob("cell_params_*.json"))
        return sorted(json_files)

    # ---------------------------
    # Preprocessing
    # ---------------------------
    def _prepare_dataframe_init(self):
        # Read dataframes.
        self.r_init_charge_df = pd.read_excel(self.file_path, sheet_name='R_init_charge').set_index('SOH/SOC')
        self.r_init_discharge_df = pd.read_excel(self.file_path, sheet_name='R_init_discharge').set_index('SOH/SOC')
        self.r1_charge_df = pd.read_excel(self.file_path, sheet_name='R1_charge').set_index('SOH/SOC')
        self.r1_discharge_df = pd.read_excel(self.file_path, sheet_name='R1_discharge').set_index('SOH/SOC')
        self.r2_charge_df = pd.read_excel(self.file_path, sheet_name='R2_charge').set_index('SOH/SOC')
        self.r2_discharge_df = pd.read_excel(self.file_path, sheet_name='R2_discharge').set_index('SOH/SOC')
        self.c1_charge_df = pd.read_excel(self.file_path, sheet_name='C1_charge').set_index('SOH/SOC')
        self.c1_discharge_df = pd.read_excel(self.file_path, sheet_name='C1_discharge').set_index('SOH/SOC')
        self.c2_charge_df = pd.read_excel(self.file_path, sheet_name='C2_charge').set_index('SOH/SOC')
        self.c2_discharge_df = pd.read_excel(self.file_path, sheet_name='C2_discharge').set_index('SOH/SOC')
        self.v_oc_df = pd.read_excel(self.file_path, sheet_name='V_oc').set_index('SOH/SOC')
        self.available_degradation_df = pd.read_excel(self.file_path, sheet_name='Degradation').set_index('Cycle')

        # Assign degradation df.
        self.degradation_df = self.available_degradation_df[self.degradation_column_name]
        self.degradation_df.columns = ['SOH']

        # 如果需要，对所有 LUT 在 SOC 轴上重采样到 [soc_min, soc_max] 上的均匀网格
        if getattr(self, "n_soc_grid_points", 0) and self.n_soc_grid_points > 0:
            # 构造新的 SOC 网格（包含 soc_min 和 soc_max）
            n_pts = max(3, int(self.n_soc_grid_points))  # 至少 3 个点，保证插值稳定
            new_soc_grid = np.linspace(self.soc_min, self.soc_max, n_pts)

            def _resample_soc_axis(df: pd.DataFrame, new_soc: np.ndarray) -> pd.DataFrame:
                df_local = df.copy()
                soh_index = df_local.index.astype(float)
                old_soc = df_local.columns.astype(float).values
                new_soc = np.asarray(new_soc, dtype=float)
                data_new = []
                for _, row in df_local.iterrows():
                    values = row.values.astype(float)
                    # 在线性插值时，对区间外做边界延拓
                    row_new = np.interp(new_soc, old_soc, values)
                    data_new.append(row_new)
                return pd.DataFrame(data_new, index=soh_index, columns=new_soc)

            self.r_init_charge_df    = _resample_soc_axis(self.r_init_charge_df,    new_soc_grid)
            self.r_init_discharge_df = _resample_soc_axis(self.r_init_discharge_df, new_soc_grid)
            self.r1_charge_df        = _resample_soc_axis(self.r1_charge_df,        new_soc_grid)
            self.r1_discharge_df     = _resample_soc_axis(self.r1_discharge_df,     new_soc_grid)
            self.r2_charge_df        = _resample_soc_axis(self.r2_charge_df,        new_soc_grid)
            self.r2_discharge_df     = _resample_soc_axis(self.r2_discharge_df,     new_soc_grid)
            self.c1_charge_df        = _resample_soc_axis(self.c1_charge_df,        new_soc_grid)
            self.c1_discharge_df     = _resample_soc_axis(self.c1_discharge_df,     new_soc_grid)
            self.c2_charge_df        = _resample_soc_axis(self.c2_charge_df,        new_soc_grid)
            self.c2_discharge_df     = _resample_soc_axis(self.c2_discharge_df,     new_soc_grid)
            self.v_oc_df             = _resample_soc_axis(self.v_oc_df,             new_soc_grid)

        # 尝试从缓存加载power limit
        cached_discharge_df, cached_charge_df = self._load_power_limits()
        
        # Build power limit dfs (如果缓存加载失败则重新计算)
        if cached_discharge_df is not None:
            self.p_max_discharge_df = cached_discharge_df
            # print("Using cached discharge power limit.")
        else:
            self.p_max_discharge_df = build_pmax_lut(
            Voc_df=self.v_oc_df,
            Rint_df=self.r_init_discharge_df,
            R1_df=self.r1_discharge_df,
            C1_df=self.c1_discharge_df,
            R2_df=self.r2_discharge_df,
            C2_df=self.c2_discharge_df,
            Qn_Ah=self.nominal_capacity_Ah,
            Vmin=0,
            Vmax=5,
            I_rated= -self.max_discharge_current,
            dt_inner=self.dt_inner_power_limit,
            T=self.dt_step_s,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
            mode="discharge",
            use_average_power=self.use_average_power_method,
            n_segments=self.n_power_segments,
            max_opt_iter=self.max_opt_iter,
            opt_tol=self.opt_tol,
            use_parallel=self.use_parallel_computation,
            n_jobs=self.n_jobs,
            )
            print("Computed discharge power limit (not using cache).")

        # self.p_max_discharge_df = - self.p_max_discharge_df

        # 创建带SOC限制检查的插值器包装
        base_interp_discharge = LUT2D(self.p_max_discharge_df)
        def p_max_discharge_interp_wrapper(soh, soc):
            if (soc < self.soc_min) or (soc > self.soc_max):
                return 0.0
            return base_interp_discharge(soh, soc)
        self.p_max_discharge_interp = p_max_discharge_interp_wrapper

        # Build charge power limit df (如果缓存加载失败则重新计算)
        if cached_charge_df is not None:
            self.p_max_charge_df = cached_charge_df
            # print("Using cached charge power limit.")
        else:
            self.p_max_charge_df = build_pmax_lut(
            Voc_df=self.v_oc_df,
            Rint_df=self.r_init_charge_df,
            R1_df=self.r1_charge_df,
            C1_df=self.c1_charge_df,
            R2_df=self.r2_charge_df,
            C2_df=self.c2_charge_df,
            Qn_Ah=self.nominal_capacity_Ah,
            Vmin=0,
            Vmax=5,
            I_rated=self.max_charge_current,
            dt_inner=self.dt_inner_power_limit,
            T=self.dt_step_s,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
            mode="charge",
            use_average_power=self.use_average_power_method,
            n_segments=self.n_power_segments,
            max_opt_iter=self.max_opt_iter,
            opt_tol=self.opt_tol,
            use_parallel=self.use_parallel_computation,
            n_jobs=self.n_jobs,
            )
            print("Computed charge power limit (not using cache).")

        # 创建带SOC限制检查的插值器包装
        base_interp_charge = LUT2D(self.p_max_charge_df)
        def p_max_charge_interp_wrapper(soh, soc):
            if (soc < self.soc_min) or (soc > self.soc_max):
                return 0.0
            return base_interp_charge(soh, soc)
        self.p_max_charge_interp = p_max_charge_interp_wrapper
        
        # 如果使用了缓存，自动保存（确保缓存是最新的）
        if cached_discharge_df is not None or cached_charge_df is not None:
            self.save_power_limits()
        # Initilize SOH and SOC.
        self.soh = self.degradation_df.loc[0]
        self.soc = self.soc_init
        self.capacity_Ah = self.nominal_capacity_Ah * self.soh
        self.efc_cycle = 0

        self.luts = {
        "Voc": LUT2D(self.v_oc_df),
        "Rint": LUT2D(self.r_init_discharge_df),
        "R1": LUT2D(self.r1_discharge_df),
        "C1": LUT2D(self.c1_discharge_df),
        "R2": LUT2D(self.r2_discharge_df),
        "C2": LUT2D(self.c2_discharge_df),
        }

    def step(self, p_request_w: float, dt_s: float | None = None) -> dict:
        if dt_s is None:
            dt_s = self.dt_step_s
        dt_s = float(dt_s)

        soc0 = self.soc

        # Add SOH
        soh = np.interp(self.efc_cycle, self.degradation_df.index, self.degradation_df)
        # cap_wh = (self.energy_max_wh - self.energy_min_wh) * soh

        # 1) 数据驱动 envelope 限制（平均功率上限）
        # 首先检查SOC是否在允许范围内
        if (soc0 < self.soc_min) or (soc0 > self.soc_max):
            # 超出SOC限制范围，功率限制为0
            p_env_lim = 0.0
            if p_request_w >= 0:
                p_env_applied = 0.0
            else:
                p_env_applied = 0.0
        else:
            if p_request_w >= 0:
                p_env_lim = self.p_max_charge_interp(soh, soc0)        # W (>=0)
                p_env_applied = float(np.clip(p_request_w, 0.0, p_env_lim))
            else:
                p_env_lim = self.p_max_discharge_interp(soh, soc0)     # W magnitude (>=0)
                p_env_applied = -float(np.clip(-p_request_w, 0.0, p_env_lim))

        # if self.soc <= self.soc_min:
        #     p_applied = 0.0
        # if self.soc >= self.soc_max:
        #     p_applied = 0.0

        # Update SOC.
        Voc0 = self.luts["Voc"](soh, soc0)
        Rint0 = self.luts["Rint"](soh, soc0)
        R10 = self.luts["R1"](soh, soc0)
        C10 = self.luts["C1"](soh, soc0)
        R20 = self.luts["R2"](soh, soc0)
        C20 = self.luts["C2"](soh, soc0)
        Qa_Ah = max(self.nominal_capacity_Ah * soh, 1e-9)
        
        R_overall = Rint0 + R10 + R20

        # disc = Voc0**2 - 4*R_overall*np.abs(p_applied)
        # disc = max(disc, 0.0)

        # I = (Voc0 - disc**0.5) / (2*R_overall)
        # 4) SOC 限制导致的"本步最大可施加功率"
        #    注意：你用的功率符号：charge为正，discharge为负
        
        # 如果envelope限制为0，直接返回0功率
        if abs(p_env_applied) < 1e-9:
            p_applied = 0.0
            I_mag = 0.0
            I_signed = 0.0
            soc = soc0
            P_soc_lim = 0.0  # 初始化P_soc_lim，避免UnboundLocalError
        elif p_env_applied < 0:  # discharge
            # 允许的最大放电电流幅值（避免 SOC 掉破下限）
            I_soc_lim_raw = max(0.0, (soc0 - self.soc_min) * Qa_Ah * 3600.0 / dt_s)
            
            # 限制电流，确保电压不会低于Vmin（Vmin=0，但实际中应该保持一个最小值）
            # 电压公式：V = Voc - I * R_overall，需要 V >= Vmin
            # 所以：I <= (Voc - Vmin) / R_overall
            Vmin_safe = 0.1  # 安全的最小电压（避免电压为负）
            I_voltage_lim = max(0.0, (Voc0 - Vmin_safe) / R_overall) if R_overall > 1e-9 else 0.0
            
            # 取SOC限制和电压限制的较小值
            I_soc_lim = min(I_soc_lim_raw, I_voltage_lim)
            
            # 由电流幅值换算最大放电"输出功率幅值"
            # 使用功率公式：P = I * (Voc - I * R)
            if I_soc_lim > 0:
                V_at_I = Voc0 - I_soc_lim * R_overall
                P_soc_lim = max(0.0, I_soc_lim * V_at_I)
            else:
                P_soc_lim = 0.0

            # 最终放电功率幅值上限 = min(envelope, soc-limit)
            P_allowed_mag = min(abs(p_env_applied), P_soc_lim)

            p_applied = -P_allowed_mag

            # 用“放电功率”反算放电电流幅值
            I_mag = current_from_power_discharge(P_allowed_mag, Voc0, R_overall)
            I_signed = +I_mag  # 放电 I>0

            soc = soc0 - (I_signed * dt_s) / (Qa_Ah * 3600.0)

        else:  # charge
            I_soc_lim = max(0.0, (self.soc_max - soc0) * Qa_Ah * 3600.0 / dt_s)
            # 由电流幅值换算最大充电“输入功率”
            P_soc_lim = max(0.0, I_soc_lim * (Voc0 + I_soc_lim * R_overall))

            P_allowed = min(p_env_applied, P_soc_lim)
            p_applied = +P_allowed

            # 用“充电功率”反算充电电流幅值
            I_mag = current_from_power_charge(P_allowed, Voc0, R_overall)
            I_signed = -I_mag  # 充电 I<0（按我们库仑更新的约定）

            soc = soc0 - (I_signed * dt_s) / (Qa_Ah * 3600.0)  # I_signed<0 => SOC↑



        # if p_applied <= 0:
        #     soc = soc0 - I * dt_s / (self.nominal_capacity_Ah * self.soh * 3600.0)
        # else:
        #     soc = soc0 + I * dt_s / (self.nominal_capacity_Ah * self.soh * 3600.0)

        self.soc = float(np.clip(soc, self.soc_min, self.soc_max))

        

        # 6) Update efc
        delta_efc = np.abs(I_mag * dt_s / (2 * self.nominal_capacity_Ah * 3600))
        self.efc_cycle += delta_efc

        # 7) Update SOH (Asuume linear relationship)
        self.soh = np.interp(self.efc_cycle, self.degradation_df.index, self.degradation_df)

        # 8) Update capacity
        self.capacity_Ah = self.nominal_capacity_Ah * self.soh

        return {
            "soc_before": soc0,
            "soc_after": self.soc,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "capacity_Ah": self.capacity_Ah,
            "dt_s": dt_s,
            "soh": self.soh,
            "efc_cycle": self.efc_cycle,
            "p_request_w": float(p_request_w),
            "p_envelope_limit_w": float(p_env_lim),
            "p_applied_w": float(p_applied),
            "p_soc_limit_w": float(P_soc_lim),
        }

    def get_step_power_limit_w(self, direction: str = "discharge", soc: float | None = None) -> float:
        """
        Cell envelope power limit (average over dt_step_s) at given (soh, soc).
        Returns a positive magnitude in W.
        """
        if soc is None:
            soc = float(self.soc)

        # 这里用当前 cell 的 soh（你 step() 里也是这样用的）
        soh = float(self.soh)

        # 检查SOC是否在允许范围内
        if (soc < self.soc_min) or (soc > self.soc_max):
            # 超出SOC限制范围，功率限制为0
            return 0.0
        
        if direction == "charge":
            return float(max(0.0, self.p_max_charge_interp(soh, soc)))
        elif direction == "discharge":
            return float(max(0.0, self.p_max_discharge_interp(soh, soc)))
        else:
            raise ValueError("direction must be 'charge' or 'discharge'")

def create_cell_from_stored_json(cache_dir, show_info=False):
    json_files = CellSLB.find_json_files(cache_dir)

    if json_files:
        # 使用第一个找到的JSON文件
        json_path = str(json_files[0])
        # print(f"Find JSON file: {json_path}")
        
        # 从JSON文件创建cell（会自动加载缓存的power limit）
        cell_from_json = CellSLB.from_json(json_path)
        
        if show_info:
            # 显示参数信息
            print(f"\nCell parameters:")
            print(f"  file_path: {cell_from_json.file_path}")
            print(f"  dt_step_s: {cell_from_json.dt_step_s}")
            print(f"  use_average_power_method: {cell_from_json.use_average_power_method}")
            print(f"  n_power_segments: {cell_from_json.n_power_segments}")
            print(f"  max_opt_iter: {cell_from_json.max_opt_iter}")
        
        return cell_from_json
    else:
        print(f"No JSON file found in {cache_dir}")
