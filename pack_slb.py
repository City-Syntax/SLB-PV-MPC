# Author: Kaifeng ZHU
# This file contains the main functions for battery pack model generation from battery cells.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


from cell_slb import CellSLB


@dataclass
class PackModelSeriesParallel:
    """
    Simple series-parallel pack model built from identical cells.

    Assumptions (simple implementation):
    - All cells are identical and balanced -> pack SOC == cell SOC.
    - Pack energy = Ns * Np * cell_energy (Wh).
    - Pack capacity = Ns * Np * cell_capacity (kWh).
    - Pack power limits derived from cell envelope:
        P_pack_limit = Ns * Np * P_cell_limit
      (because pack power = sum over all cells; each cell sees same per-cell power
       when current is shared equally among parallel strings and voltage sums in series)
    """

    cell: CellSLB
    Ns: int = 1
    Np: int = 1

    def __post_init__(self):
        if self.Ns <= 0 or self.Np <= 0:
            raise ValueError("Ns and Np must be positive integers.")

    # --------- Basic properties ----------
    @property
    def soc(self) -> float:
        return self.cell.soc

    @property
    def soc_min(self) -> float:
        return self.cell.soc_min

    @property
    def soc_max(self) -> float:
        return self.cell.soc_max

    @property
    def capacity_Ah(self) -> float:
        # pack total usable capacity (kWh)
        return float(self.Ns * self.Np * self.cell.capacity_Ah)

    # @property
    # def stored_energy_kwh(self) -> float:
    #     # pack stored energy (kWh), based on cell current energy_wh (offset included)
    #     return float(self.Ns * self.Np * self.cell.stored_energy_kwh)

    @property
    def soh(self) -> float:
        return self.cell.soh

    @property
    def nominal_capacity_Ah(self) -> float:
        return float(self.Ns * self.Np * self.cell.nominal_capacity_Ah)

    def set_soc(self, soc: float):
        self.cell.soc = soc

    # --------- Limits ----------
    def get_step_power_limit_w(self, direction: str = "discharge", soc: float | None = None) -> float:
        """
        Pack power limit based on cell envelope at given SOC:
          P_pack_max = Ns * Np * P_cell_max
        direction:
          "charge" returns +limit (W)
          "discharge" returns +limit magnitude (W)
        """
        p_cell_lim = self.cell.get_step_power_limit_w(soc=soc, direction=direction)
        return float(self.Ns * self.Np * p_cell_lim)

    # --------- Step simulation ----------
    def step(self, p_request_pack_w: float, dt_s: float | None = None) -> dict:
        """
        Apply pack power request for one step.
        We convert to per-cell power request:
          p_cell_request = p_pack_request / (Ns * Np)
        because pack power is sum of cell powers across all Ns*Np cells.

        Then call cell.step() once (balanced assumption),
        and scale applied power back to pack:
          p_pack_applied = p_cell_applied * Ns * Np
        """
        if dt_s is None:
            dt_s = self.cell.dt_step_s
        dt_s = float(dt_s)

        n_cells = self.Ns * self.Np
        p_cell_request = float(p_request_pack_w) / n_cells

        out_cell = self.cell.step(p_request_w=p_cell_request, dt_s=dt_s)

        p_cell_applied = float(out_cell["p_applied_w"])
        p_pack_applied = p_cell_applied * n_cells

        # also scale limits for reporting
        # (cell returns envelope limit as magnitude for discharge; we keep same convention)
        p_env_cell = float(out_cell.get("p_envelope_limit_w", np.nan))
        p_soc_cell = float(out_cell.get("p_soc_limit_w", np.nan))

        # If the request was discharge (negative), cell's p_envelope_limit_w is magnitude.
        # Pack equivalent magnitude:
        p_env_pack = p_env_cell * n_cells if np.isfinite(p_env_cell) else np.nan
        p_soc_pack = p_soc_cell * n_cells if np.isfinite(p_soc_cell) else np.nan

        return {
            "Ns": self.Ns,
            "Np": self.Np,
            "n_cells": n_cells,
            "soc_before": out_cell["soc_before"],
            "soc_after": out_cell["soc_after"],
            "soc_min": out_cell.get("soc_min", self.soc_min),
            "soc_max": out_cell.get("soc_max", self.soc_max),

            "p_request_pack_w": float(p_request_pack_w),
            "p_applied_pack_w": float(p_pack_applied),

            # for debugging / analysis
            "p_request_cell_w": float(p_cell_request),
            "p_applied_cell_w": float(p_cell_applied),

            "p_envelope_limit_pack_w": float(p_env_pack),
            "p_soc_limit_pack_w": float(p_soc_pack),
            "dt_s": dt_s,
        }


# ---------------------------
# Example usage
# ---------------------------
# if __name__ == "__main__":
#     df = pd.read_csv("sample_cell.csv")
#     cell = CellModelDataDriven(df, dt_step_s=300, soc_init=0.8, soc_min=0.1, soc_max=0.9)

#     pack = PackModelSeriesParallel(cell=cell, Ns=14, Np=10)  # example: 14s10p

#     print("Pack capacity (kWh):", pack.capacity_kwh)
#     print("Pack SOC:", pack.soc)
#     print("Pack discharge limit at current SOC (W mag):", pack.get_step_power_limit_w("discharge"))
#     print("Pack charge limit at current SOC (W):", pack.get_step_power_limit_w("charge"))

#     out = pack.step(p_request_pack_w=-2000.0, dt_s=300)  # request -2 kW discharge for 5 min
#     print(out)
