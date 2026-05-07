# Author: Kaifeng ZHU
# This file contains the main functions for battery system model generation from battery packs.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pack_slb import PackModelSeriesParallel

@dataclass
class BatterySystemSeriesParallel:
    """
    Battery system composed of identical packs in series-parallel.

    Topology:
      - N_series: packs in series per string
      - N_parallel: number of parallel strings
      - Total packs = N_series * N_parallel

    Assumption (simple & control-friendly):
      - packs are identical and balanced -> system SOC == pack SOC
      - requested system power is shared equally across all packs
      - system energy/capacity/power limits scale linearly with number of packs
    """
    pack: PackModelSeriesParallel
    N_series: int = 1
    N_parallel: int = 1

    def __post_init__(self):
        if self.N_series <= 0 or self.N_parallel <= 0:
            raise ValueError("N_series and N_parallel must be positive integers.")

    @property
    def n_packs(self) -> int:
        return int(self.N_series * self.N_parallel)

    @property
    def soc(self) -> float:
        return self.pack.soc

    @property
    def capacity_Ah(self) -> float:
        return float(self.n_packs * self.pack.capacity_Ah)

    # @property
    # def stored_energy_kwh(self) -> float:
    #     return float(self.n_packs * self.pack.stored_energy_kwh)
    
    @property
    def soh(self) -> float:
        return self.pack.soh

    @property
    def nominal_capacity_Ah(self) -> float:
        return float(self.n_packs * self.pack.nominal_capacity_Ah)

    def set_soc(self, soc: float):
        self.pack.set_soc(soc)

    # --------- Power limits (kW) ----------
    def get_step_power_limit_kw(self, direction: str = "discharge", soc: float | None = None) -> float:
        """
        System power limit derived from pack limits:
          P_sys_max = n_packs * P_pack_max
        direction:
          "charge" -> positive limit (kW)
          "discharge" -> positive magnitude limit (kW)
        """
        p_pack_lim_w = self.pack.get_step_power_limit_w(direction=direction, soc=soc)
        p_sys_lim_kw = (self.n_packs * p_pack_lim_w) / 1000.0
        return float(p_sys_lim_kw)

    # --------- Step (system-level, kW) ----------
    def step(self, p_request_sys_kw: float, dt_s: float | None = None) -> dict:
        """
        Apply system power request in kW for one step.
        Convert to per-pack request:
          P_pack_request = P_sys / n_packs
        Call pack.step once (balanced assumption), then scale applied back.
        """
        if dt_s is None:
            dt_s = self.pack.cell.dt_step_s
        dt_s = float(dt_s)

        p_request_sys_w = float(p_request_sys_kw) * 1000.0
        p_request_pack_w = p_request_sys_w / self.n_packs

        out_pack = self.pack.step(p_request_pack_w=p_request_pack_w, dt_s=dt_s)
        p_applied_sys_w = float(out_pack["p_applied_pack_w"]) * self.n_packs

        p_envelope_limit_system_kw = float(out_pack["p_envelope_limit_pack_w"]) * self.n_packs / 1000.0
        p_soc_limit_system_kw = float(out_pack["p_soc_limit_pack_w"]) * self.n_packs / 1000.0

        return {
            "N_series": self.N_series,
            "N_parallel": self.N_parallel,
            "n_packs": self.n_packs,

            "soc_before": out_pack["soc_before"],
            "soc_after": out_pack["soc_after"],

            "capacity_Ah": self.capacity_Ah,


            "p_request_sys_kw": float(p_request_sys_kw),
            "p_applied_sys_kw": float(p_applied_sys_w / 1000.0),

            "p_envelope_limit_sys_kw": float(p_envelope_limit_system_kw),
            "p_soc_limit_sys_kw": float(p_soc_limit_system_kw),

            # "p_request_pack_kw": float(p_request_pack_w / 1000.0),
            # "p_applied_pack_kw": float(out_pack["p_applied_pack_w"] / 1000.0),

            "dt_s": dt_s,
        }
