# Author: Kaifeng ZHU
# This file contains the main functions for rule-based with arbitrage control calculation (RBC-A).
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from battery_system_slb import BatterySystemSeriesParallel


@dataclass
class RuleBasedCostConfig:
    """
    Price-based arbitrage thresholds.

    - price_low: if price <= price_low, allow charging from grid (if SOC < soc_max)
    - price_high: if price >= price_high, allow discharging to grid (if SOC > soc_min)
    """

    # Static thresholds (used when not using top-k schedule)
    price_low: float = 0.05
    price_high: float = 0.20

    # Day-ahead schedule mode: charge in k lowest-price hours, discharge in k highest-price hours
    top_k_hours: int = 4


class RuleBasedCostController:
    """
    Rule-based controller for PV-battery-building power management with simple arbitrage.

    Baseline (same as RuleBasedController):
      - p_excess = PV - Demand
      - If p_excess > 0: charge battery (if SOC < SOC_max), export remainder
      - If p_excess < 0: discharge battery (if SOC > SOC_min), import remainder

    Added arbitrage:
      - If Price >= price_high:
          After prioritizing building demand, discharge to grid for profit if SOC > SOC_min.
          (This means battery can discharge more than the demand deficit to create export.)
      - If Price <= price_low:
          Charge from grid if SOC < SOC_max (even if PV is low / demand is high).

    Sign convention:
      p_batt_kw > 0 : charge
      p_batt_kw < 0 : discharge
    """

    def __init__(
        self,
        system: BatterySystemSeriesParallel,
        cfg: RuleBasedCostConfig | None = None,
        *,
        dt_s: float | None = None,
    ):
        self.system = system
        self.cfg = cfg if cfg is not None else RuleBasedCostConfig()

        self.dt_s = float(dt_s) if dt_s is not None else float(system.pack.cell.dt_step_s)
        self.dt_h = self.dt_s / 3600.0

        self.soc_min = float(system.pack.cell.soc_min)
        self.soc_max = float(system.pack.cell.soc_max)

    def power_limits_kw(self, soc: float) -> Tuple[float, float]:
        """
        Returns (p_ch_max, p_dis_max) in kW, both positive magnitudes.
        """
        soc = float(np.clip(float(soc), self.soc_min, self.soc_max))
        p_ch = float(self.system.get_step_power_limit_kw(direction="charge", soc=soc))
        p_dis = float(self.system.get_step_power_limit_kw(direction="discharge", soc=soc))
        return p_ch, p_dis

    def compute_action(
        self,
        demand_kw: float,
        pv_kw: float,
        price: float,
        soc: float,
        *,
        allow_charge: bool = False,
        allow_discharge: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Returns (p_batt_request_kw, p_export_kw, p_import_kw).
        """
        demand_kw = float(demand_kw)
        pv_kw = float(pv_kw)
        price = float(price)
        soc = float(soc)

        p_ch_max, p_dis_max = self.power_limits_kw(soc)

        # Net building load after PV (positive => deficit, negative => surplus)
        p_net = demand_kw - pv_kw

        # Default: no battery action
        p_batt_req = 0.0

        # -------- Arbitrage overrides --------
        # Priority 1: explicit schedule (top-k hours)
        if allow_discharge:
            if soc > self.soc_min and p_dis_max > 0:
                p_batt_req = -p_dis_max
            else:
                p_batt_req = 0.0

        elif allow_charge:
            if soc < self.soc_max and p_ch_max > 0:
                p_batt_req = +p_ch_max
            else:
                p_batt_req = 0.0

        # Priority 2: static threshold-based arbitrage
        elif price >= float(self.cfg.price_high):
            # High price: discharge as much as possible (if allowed) to avoid import and create export.
            if soc > self.soc_min and p_dis_max > 0:
                p_batt_req = -p_dis_max
            else:
                p_batt_req = 0.0

        elif price <= float(self.cfg.price_low):
            # Low price: charge from grid as much as possible (if allowed).
            if soc < self.soc_max and p_ch_max > 0:
                p_batt_req = +p_ch_max
            else:
                p_batt_req = 0.0

        else:
            # -------- Baseline self-consumption logic --------
            p_excess = pv_kw - demand_kw
            if p_excess > 0:
                # Surplus: try to charge from PV
                if soc < self.soc_max and p_ch_max > 0:
                    p_batt_req = min(p_excess, p_ch_max)
                else:
                    p_batt_req = 0.0
            elif p_excess < 0:
                # Deficit: try to discharge to cover building
                p_deficit = -p_excess
                if soc > self.soc_min and p_dis_max > 0:
                    p_batt_req = -min(p_deficit, p_dis_max)
                else:
                    p_batt_req = 0.0
            else:
                p_batt_req = 0.0

        # Compute implied grid import/export with requested action
        p_grid_req = p_net + p_batt_req  # demand - pv + p_batt
        p_import = max(p_grid_req, 0.0)
        p_export = max(-p_grid_req, 0.0)

        return float(p_batt_req), float(p_export), float(p_import)

    def run(
        self,
        df: pd.DataFrame,
        *,
        col_demand: str = "Demand",
        col_pv: str = "PV",
        col_price: str = "Price",
        time_col: str = "datetime_utc",
        use_topk_schedule: bool = True,
        top_k_hours: int | None = None,
        efc_retire: float | None = None,
    ) -> pd.DataFrame:
        """
        Run rule-based control with optional day-ahead top-k schedule.
        
        Args:
            df: DataFrame with simulation data
            col_demand: Column name for building demand (kW)
            col_pv: Column name for PV generation (kW)
            col_price: Column name for electricity price
            time_col: Column name for datetime
            use_topk_schedule: If True, charge in the lowest-k price hours and discharge in the highest-k price hours
                               (computed per day from the price profile).
            top_k_hours: override cfg.top_k_hours
            efc_retire: if not None, retire the battery once effective full cycles
                        reach this value; subsequent timesteps are simulated as
                        a no-battery baseline (p_batt = 0).
        """
        out = df.copy()
        n = len(out)
        if n == 0:
            return out

        # Ensure datetime column exists and is datetime type (needed for schedule)
        if time_col not in out.columns:
            datetime_cols = [col for col in out.columns if "datetime" in col.lower() or "time" in col.lower()]
            if datetime_cols:
                time_col = datetime_cols[0]
            else:
                raise ValueError("Could not find datetime column. Please specify time_col.")

        if not pd.api.types.is_datetime64_any_dtype(out[time_col]):
            out[time_col] = pd.to_datetime(out[time_col])

        # Precompute per-day schedule if enabled
        k = int(top_k_hours) if top_k_hours is not None else int(self.cfg.top_k_hours)
        k = max(0, k)

        allow_charge_arr = np.zeros(n, dtype=bool)
        allow_discharge_arr = np.zeros(n, dtype=bool)

        if use_topk_schedule and k > 0:
            # For each date, find the k lowest/highest price hours (ties resolved by stable sort)
            tmp = out[[time_col, col_price]].copy()
            tmp["date"] = tmp[time_col].dt.date
            tmp["hour"] = tmp[time_col].dt.hour

            # In case there are multiple entries per hour, use mean price per (date, hour)
            hourly = (
                tmp.groupby(["date", "hour"], as_index=False)[col_price]
                .mean()
                .rename(columns={col_price: "price_hour"})
            )

            # Build sets: date -> {hours}
            low_hours_by_date: dict = {}
            high_hours_by_date: dict = {}

            for date, g in hourly.groupby("date"):
                g = g.sort_values(["price_hour", "hour"], ascending=[True, True])
                low_hours = g["hour"].head(min(k, len(g))).tolist()
                g2 = g.sort_values(["price_hour", "hour"], ascending=[False, True])
                high_hours = g2["hour"].head(min(k, len(g2))).tolist()

                low_hours_by_date[date] = set(int(h) for h in low_hours)
                high_hours_by_date[date] = set(int(h) for h in high_hours)

            # Map each row to schedule
            dates = out[time_col].dt.date.to_numpy()
            hours = out[time_col].dt.hour.to_numpy()
            for i in range(n):
                d = dates[i]
                h = int(hours[i])
                allow_charge_arr[i] = h in low_hours_by_date.get(d, set())
                allow_discharge_arr[i] = h in high_hours_by_date.get(d, set())

        p_req = np.zeros(n, dtype=float)
        p_app = np.zeros(n, dtype=float)
        soc = np.zeros(n, dtype=float)
        p_grid = np.zeros(n, dtype=float)
        p_env_lim = np.zeros(n, dtype=float)

        import_kwh = np.zeros(n, dtype=float)
        export_kwh = np.zeros(n, dtype=float)
        cost = np.zeros(n, dtype=float)

        efc = np.zeros(n, dtype=float)
        soh = np.zeros(n, dtype=float)

        retired = False
        last_efc = 0.0
        last_soh = float(self.system.pack.cell.soh)

        for t in range(n):
            d_real = float(out[col_demand].iloc[t])
            pv_real = float(out[col_pv].iloc[t])
            price_t = float(out[col_price].iloc[t]) if col_price in out.columns else 0.0

            if not retired:
                soc_t = float(self.system.soc)

                a_req, _, _ = self.compute_action(
                    d_real,
                    pv_real,
                    price_t,
                    soc_t,
                    allow_charge=bool(allow_charge_arr[t]),
                    allow_discharge=bool(allow_discharge_arr[t]),
                )

                # envelope limit at current SOC for logging
                p_ch_env, p_dis_env = self.power_limits_kw(soc_t)
                p_env_lim[t] = p_ch_env if a_req >= 0 else p_dis_env

                step_out = self.system.step(p_request_sys_kw=float(a_req), dt_s=self.dt_s)

                p_req[t] = float(a_req)
                p_app[t] = float(step_out["p_applied_sys_kw"])
                soc[t] = float(step_out["soc_after"])

                p_grid_t = d_real - pv_real + p_app[t]
                p_grid[t] = p_grid_t

                import_kwh[t] = max(p_grid_t, 0.0) * self.dt_h
                export_kwh[t] = max(-p_grid_t, 0.0) * self.dt_h

                # buy and sell price are assumed equal
                cost[t] = price_t * (import_kwh[t] - export_kwh[t])

                efc_t = float(self.system.pack.cell.efc_cycle)
                soh_t = float(self.system.pack.cell.soh)
                efc[t] = efc_t
                soh[t] = soh_t
                last_efc = efc_t
                last_soh = soh_t

                # If retirement threshold is set and reached, retire from next step
                if efc_retire is not None and efc_t >= float(efc_retire):
                    retired = True
            else:
                # Battery retired: behave as no-battery baseline (p_batt = 0).
                p_req[t] = 0.0
                p_app[t] = 0.0
                p_env_lim[t] = 0.0

                # SOC, EFC, SOH stay at last values for logging
                soc[t] = soc[t - 1] if t > 0 else float(self.system.soc)

                p_grid_t = d_real - pv_real
                p_grid[t] = p_grid_t

                import_kwh[t] = max(p_grid_t, 0.0) * self.dt_h
                export_kwh[t] = max(-p_grid_t, 0.0) * self.dt_h

                cost[t] = price_t * (import_kwh[t] - export_kwh[t])

                efc[t] = last_efc
                soh[t] = last_soh

        out["p_batt_request_kw"] = p_req
        out["p_batt_applied_kw"] = p_app
        out["soc"] = soc
        out["p_grid_kw"] = p_grid
        out["p_env_limit_sys_kw"] = p_env_lim

        out["grid_import_kwh"] = import_kwh
        out["grid_export_kwh"] = export_kwh
        out["grid_cost"] = cost
        out["grid_cost_cum"] = np.cumsum(cost)

        out["efc_cycle"] = efc
        out["soh"] = soh

        if use_topk_schedule and k > 0:
            out["allow_charge_topk"] = allow_charge_arr.astype(int)
            out["allow_discharge_topk"] = allow_discharge_arr.astype(int)
            out["topk_k_hours"] = k

        return out


