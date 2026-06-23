"""
piping_model.py
-----------------
Single-line piping network pressure-drop calculator built on CoolProp
fluid properties and the Colebrook-White friction-factor correlation,
plus a classic "economic pipe diameter" optimization (minimize
annualized capital + pumping-energy cost) using scipy.optimize.

Units: SI throughout (m, Pa, kg/s or m3/s, K) unless noted.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from CoolProp.CoolProp import PropsSI

C_TO_K = 273.15
G = 9.80665


def fluid_props(fluid, T_C, P_Pa):
    """Return (density kg/m3, dynamic viscosity Pa.s) from CoolProp."""
    T_K = T_C + C_TO_K
    rho = PropsSI("D", "T", T_K, "P", P_Pa, fluid)
    mu = PropsSI("V", "T", T_K, "P", P_Pa, fluid)
    return rho, mu


def darcy_friction_factor(Re, rel_roughness):
    """
    Darcy (Moody) friction factor.
    Laminar (Re < 2300): f = 64/Re
    Turbulent: Colebrook-White, solved exactly with brentq.
    """
    if Re <= 0:
        raise ValueError("Reynolds number must be positive.")
    if Re < 2300:
        return 64.0 / Re

    def colebrook(f):
        return 1.0 / np.sqrt(f) + 2.0 * np.log10(
            rel_roughness / 3.7 + 2.51 / (Re * np.sqrt(f))
        )

    return brentq(colebrook, 1e-5, 1.0, xtol=1e-10)


@dataclass
class PipeSegment:
    name: str
    length_m: float
    diameter_m: float
    roughness_m: float = 4.5e-5      # commercial steel, m
    elevation_change_m: float = 0.0  # outlet - inlet
    k_fittings: float = 0.0          # sum of minor-loss K coefficients


@dataclass
class SegmentResult:
    name: str
    velocity_ms: float
    reynolds: float
    friction_factor: float
    dp_friction_pa: float
    dp_minor_pa: float
    dp_elevation_pa: float
    dp_total_pa: float


def evaluate_network(fluid, T_C, P_Pa, volumetric_flow_m3s,
                      segments: List[PipeSegment]):
    """
    Evaluate a series piping network (single flow path) and return
    per-segment results plus total pressure drop.
    """
    rho, mu = fluid_props(fluid, T_C, P_Pa)
    results = []
    total_dp = 0.0
    for seg in segments:
        A = np.pi / 4.0 * seg.diameter_m ** 2
        v = volumetric_flow_m3s / A
        Re = rho * v * seg.diameter_m / mu
        rel_rough = seg.roughness_m / seg.diameter_m
        f = darcy_friction_factor(Re, rel_rough)
        dp_fric = f * (seg.length_m / seg.diameter_m) * (rho * v ** 2 / 2.0)
        dp_minor = seg.k_fittings * (rho * v ** 2 / 2.0)
        dp_elev = rho * G * seg.elevation_change_m
        dp_seg = dp_fric + dp_minor + dp_elev
        total_dp += dp_seg
        results.append(SegmentResult(
            name=seg.name, velocity_ms=v, reynolds=Re, friction_factor=f,
            dp_friction_pa=dp_fric, dp_minor_pa=dp_minor,
            dp_elevation_pa=dp_elev, dp_total_pa=dp_seg,
        ))
    return results, total_dp, rho, mu


@dataclass
class EconomicDiameterInputs:
    fluid: str
    T_C: float
    P_Pa: float
    volumetric_flow_m3s: float
    length_m: float
    roughness_m: float
    elevation_change_m: float
    k_fittings: float
    pump_efficiency: float
    electricity_price_per_kWh: float
    operating_hours_per_year: float
    capital_cost_coeff: float   # $ per m length per (m diameter)^exponent
    capital_cost_exponent: float
    capital_recovery_factor: float  # e.g. CRF = i(1+i)^n / ((1+i)^n - 1)
    d_min_m: float = 0.01
    d_max_m: float = 1.5


def _segment_for_diameter(inp: EconomicDiameterInputs, D):
    return PipeSegment(
        name="economic_segment", length_m=inp.length_m, diameter_m=D,
        roughness_m=inp.roughness_m,
        elevation_change_m=inp.elevation_change_m,
        k_fittings=inp.k_fittings,
    )


def annualized_cost(inp: EconomicDiameterInputs, D):
    """Total annualized cost ($/yr) for a given pipe diameter D (m)."""
    seg = _segment_for_diameter(inp, D)
    results, total_dp, rho, mu = evaluate_network(
        inp.fluid, inp.T_C, inp.P_Pa, inp.volumetric_flow_m3s, [seg]
    )
    pump_power_W = inp.volumetric_flow_m3s * total_dp / inp.pump_efficiency
    pump_power_kW = pump_power_W / 1000.0
    annual_energy_cost = (pump_power_kW * inp.operating_hours_per_year
                           * inp.electricity_price_per_kWh)
    capital_cost = (inp.capital_cost_coeff * inp.length_m
                     * D ** inp.capital_cost_exponent)
    annualized_capital = capital_cost * inp.capital_recovery_factor
    total = annualized_capital + annual_energy_cost
    return {
        "diameter_m": D,
        "velocity_ms": results[0].velocity_ms,
        "reynolds": results[0].reynolds,
        "friction_factor": results[0].friction_factor,
        "dp_total_pa": total_dp,
        "pump_power_kW": pump_power_kW,
        "annual_energy_cost": annual_energy_cost,
        "capital_cost": capital_cost,
        "annualized_capital": annualized_capital,
        "total_annualized_cost": total,
    }


def optimize_economic_diameter(inp: EconomicDiameterInputs):
    """
    Minimize total annualized cost (capital + pumping energy) over pipe
    diameter using a bounded 1-D scipy.optimize routine.
    """
    def obj(D):
        return annualized_cost(inp, D)["total_annualized_cost"]

    result = minimize_scalar(
        obj, bounds=(inp.d_min_m, inp.d_max_m), method="bounded",
        options={"xatol": 1e-6},
    )
    detail = annualized_cost(inp, result.x)
    return detail, result
