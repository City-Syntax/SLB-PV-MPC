# Author: Kaifeng ZHU
# This file contains the main functions for rule-based control calculation (RBC).
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

from battery_system_slb import BatterySystemSeriesParallel


class RuleBasedController:
    """
    Rule-based controller for PV-battery-building power management.
    
    Implements the flowchart logic:
    1. Calculate excess power: P_excess = PV - Demand
    2. If P_excess > 0 (surplus):
       - Charge battery if SoC < SoC_max, export remainder
    3. If P_excess <= 0 (deficit):
       - Discharge battery if SoC > SoC_min, import remainder
    
    Sign convention:
      p_batt_kw > 0 : charge
      p_batt_kw < 0 : discharge
    """

    def __init__(self, system: BatterySystemSeriesParallel, dt_s: float | None = None):
        """
        Initialize rule-based controller.
        
        Args:
            system: BatterySystemSeriesParallel instance
            dt_s: Time step in seconds (defaults to system.pack.cell.dt_step_s)
        """
        self.system = system
        
        # Set time step
        if dt_s is not None:
            self.dt_s = float(dt_s)
        else:
            self.dt_s = float(system.pack.cell.dt_step_s)
        self.dt_h = self.dt_s / 3600.0
        
        # Extract SOC limits from system
        self.soc_min = float(system.pack.cell.soc_min)
        self.soc_max = float(system.pack.cell.soc_max)

    def power_limits_kw(self, soc: float) -> Tuple[float, float]:
        """
        Get power limits at given SOC.
        
        Args:
            soc: State of charge (0-1)
            
        Returns:
            (p_ch_max, p_dis_max) in kW (both positive)
        """
        # Ensure SOC is within valid range
        soc = float(np.clip(soc, self.soc_min, self.soc_max))
        
        # Get power limits from system
        # Note: system.get_step_power_limit_kw uses the current SOH of the cell
        p_ch = float(self.system.get_step_power_limit_kw(direction="charge", soc=soc))
        p_dis = float(self.system.get_step_power_limit_kw(direction="discharge", soc=soc))
        
        return p_ch, p_dis
    
    def debug_power_limits(self, soc: float) -> dict:
        """
        Debug method to check power limits and system state.
        
        Args:
            soc: State of charge (0-1)
            
        Returns:
            Dictionary with debug information
        """
        soc = float(np.clip(soc, self.soc_min, self.soc_max))
        cell = self.system.pack.cell
        
        # Get current SOH
        soh = float(cell.soh)
        
        # Try to get power limits directly from cell
        p_ch_cell_w = cell.get_step_power_limit_w(direction="charge", soc=soc)
        p_dis_cell_w = cell.get_step_power_limit_w(direction="discharge", soc=soc)
        
        # Get from system level
        p_ch_sys_kw = self.system.get_step_power_limit_kw(direction="charge", soc=soc)
        p_dis_sys_kw = self.system.get_step_power_limit_kw(direction="discharge", soc=soc)
        
        return {
            "soc": soc,
            "soh": soh,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "p_ch_cell_w": p_ch_cell_w,
            "p_dis_cell_w": p_dis_cell_w,
            "p_ch_sys_kw": p_ch_sys_kw,
            "p_dis_sys_kw": p_dis_sys_kw,
            "n_packs": self.system.n_packs,
            "pack_Ns": self.system.pack.Ns,
            "pack_Np": self.system.pack.Np,
        }

    def compute_action(self, demand_kw: float, pv_kw: float, soc: float) -> Tuple[float, float, float]:
        """
        Compute battery action based on rule-based control logic.
        
        Implements the flowchart:
        1. Calculate P_excess = PV - Demand
        2. If P_excess > 0: charge battery, export remainder
        3. If P_excess <= 0: discharge battery, import remainder
        
        Args:
            demand_kw: Building demand in kW
            pv_kw: PV generation in kW
            soc: Current state of charge (0-1)
            
        Returns:
            (p_batt_request_kw, p_export_kw, p_import_kw)
            - p_batt_request_kw: Battery power request (positive=charge, negative=discharge)
            - p_export_kw: Power exported to grid (positive)
            - p_import_kw: Power imported from grid (positive)
        """
        demand_kw = float(demand_kw)
        pv_kw = float(pv_kw)
        soc = float(soc)
        
        # Step 1: Calculate excess power
        p_excess = pv_kw - demand_kw
        
        # Get power limits at current SOC
        p_ch_max, p_dis_max = self.power_limits_kw(soc)
        
        # Initialize outputs
        p_batt_request_kw = 0.0
        p_export_kw = 0.0
        p_import_kw = 0.0
        
        # Step 2: Decision tree based on flowchart
        if p_excess > 0:
            # Surplus: try to charge battery
            if soc < self.soc_max and p_ch_max > 0:
                # Battery can be charged (check both SOC and power limit)
                if p_excess > p_ch_max:
                    # Excess exceeds max charge power
                    p_batt_request_kw = p_ch_max
                    p_export_kw = p_excess - p_ch_max
                else:
                    # All excess goes to battery
                    p_batt_request_kw = p_excess
                    p_export_kw = 0.0
            else:
                # Battery cannot charge (full or no power capacity), export all excess
                p_batt_request_kw = 0.0
                p_export_kw = p_excess
        elif p_excess < 0:
            # Deficit: try to discharge battery
            p_deficit = abs(p_excess)  # Positive deficit amount
            
            if soc > self.soc_min and p_dis_max > 0:
                # Battery can be discharged (check both SOC and power limit)
                if p_deficit > p_dis_max:
                    # Deficit exceeds max discharge power
                    p_batt_request_kw = -p_dis_max  # Negative for discharge
                    p_import_kw = p_deficit - p_dis_max
                else:
                    # Battery can cover all deficit
                    p_batt_request_kw = -p_deficit  # Negative for discharge
                    p_import_kw = 0.0
            else:
                # Battery cannot discharge (empty or no power capacity), import all deficit
                p_batt_request_kw = 0.0
                p_import_kw = p_deficit
        else:
            # p_excess == 0: No excess, no deficit, no battery action
            p_batt_request_kw = 0.0
            p_export_kw = 0.0
            p_import_kw = 0.0
        
        return p_batt_request_kw, p_export_kw, p_import_kw

    def run(
        self,
        df: pd.DataFrame,
        *,
        col_demand: str = "Demand",
        col_pv: str = "PV",
        col_price: str = "Price",
    ) -> pd.DataFrame:
        """
        Run rule-based control simulation.
        
        Args:
            df: DataFrame with columns: Demand, PV, Price
            col_demand: Column name for building demand (kW)
            col_pv: Column name for PV generation (kW)
            col_price: Column name for electricity price
            
        Returns:
            DataFrame with original columns plus:
            - p_batt_request_kw: Requested battery power
            - p_batt_applied_kw: Actually applied battery power
            - soc: State of charge after step
            - p_grid_kw: Grid power (positive=import, negative=export)
            - p_env_limit_sys_kw: Envelope power limit at current SOC
            - grid_import_kwh: Energy imported from grid
            - grid_export_kwh: Energy exported to grid
            - grid_cost: Cost for this timestep
            - grid_cost_cum: Cumulative cost
            - efc_cycle: Equivalent full cycles
            - soh: State of health
        """
        out = df.copy()
        n = len(out)
        if n == 0:
            return out

        # Initialize output arrays
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

        for t in range(n):
            # Get current SOC
            soc_t = float(self.system.soc)
            
            # Get current values
            d_real = float(out[col_demand].iloc[t])
            pv_real = float(out[col_pv].iloc[t])
            price_t = float(out[col_price].iloc[t]) if col_price in out.columns else 0.0
            
            # Compute action using rule-based logic
            p_batt_req, p_export, p_import = self.compute_action(d_real, pv_real, soc_t)
            
            # Get envelope limit at current SOC for logging
            p_ch_env, p_dis_env = self.power_limits_kw(soc_t)
            p_env_lim[t] = p_ch_env if p_batt_req >= 0 else p_dis_env
            
            # Apply action to battery system
            step_out = self.system.step(p_request_sys_kw=float(p_batt_req), dt_s=self.dt_s)
            
            # Record battery metrics
            p_req[t] = float(p_batt_req)
            p_app[t] = float(step_out["p_applied_sys_kw"])
            soc[t] = float(step_out["soc_after"])
            
            # Calculate grid power: demand - pv + battery
            # Positive = import, negative = export
            p_grid_t = d_real - pv_real + p_app[t]
            p_grid[t] = p_grid_t
            
            # Calculate energy import/export
            import_kwh[t] = max(p_grid_t, 0.0) * self.dt_h
            export_kwh[t] = max(-p_grid_t, 0.0) * self.dt_h
            
            # Calculate cost (assuming same buy/sell price)
            cost[t] = price_t * (import_kwh[t] - export_kwh[t])
            
            # Record degradation metrics
            efc[t] = float(self.system.pack.cell.efc_cycle)
            soh[t] = float(self.system.pack.cell.soh)

        # Add output columns to DataFrame
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

        return out

