"""
cycle_model.py
----------------
Vapor-compression cycle (heat pump / refrigeration) thermodynamic model
built on CoolProp, plus a constrained scipy.optimize routine that finds
the evaporating / condensing saturation temperatures that maximize COP
subject to realistic engineering constraints (heat-exchanger approach
temperatures and a compressor discharge-temperature limit).

All temperatures in this module's public API are in degrees Celsius
unless noted "_K". All pressures are in Pa, enthalpies in J/kg.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from CoolProp.CoolProp import PropsSI

C_TO_K = 273.15

REFRIGERANTS = [
    "R134a", "R410A", "R32", "R290", "R600a",
    "R1234yf", "R407C", "AMMONIA", "R744",
]


@dataclass
class CycleResult:
    fluid: str
    T_evap_C: float
    T_cond_C: float
    superheat_K: float
    subcooling_K: float
    eta_isentropic: float
    P_evap_Pa: float
    P_cond_Pa: float
    T1_C: float          # compressor inlet (after superheat)
    T2_C: float          # compressor outlet (actual, after inefficiency)
    T2s_C: float         # compressor outlet (isentropic)
    T3_C: float          # condenser outlet (after subcooling)
    T4_C: float          # evaporator inlet (after expansion valve)
    h1: float
    h2: float
    h2s: float
    h3: float
    h4: float
    q_evap: float        # specific refrigerating/heating effect, J/kg
    q_cond: float
    w_comp: float
    cop_heating: float
    cop_cooling: float
    pressure_ratio: float

    def to_dict(self):
        return {
            "Fluid": self.fluid,
            "T_evap (C)": round(self.T_evap_C, 2),
            "T_cond (C)": round(self.T_cond_C, 2),
            "P_evap (kPa)": round(self.P_evap_Pa / 1000, 1),
            "P_cond (kPa)": round(self.P_cond_Pa / 1000, 1),
            "Pressure ratio": round(self.pressure_ratio, 2),
            "Compressor suction T (C)": round(self.T1_C, 1),
            "Compressor discharge T (C)": round(self.T2_C, 1),
            "Subcooled liquid T (C)": round(self.T3_C, 1),
            "q_evap (kJ/kg)": round(self.q_evap / 1000, 2),
            "q_cond (kJ/kg)": round(self.q_cond / 1000, 2),
            "w_comp (kJ/kg)": round(self.w_comp / 1000, 2),
            "COP (heating)": round(self.cop_heating, 3),
            "COP (cooling)": round(self.cop_cooling, 3),
        }


def solve_cycle(fluid, T_evap_C, T_cond_C, superheat_K, subcooling_K,
                 eta_isentropic):
    """
    Solve a single-stage vapor-compression cycle.

    Parameters
    ----------
    fluid : str
        CoolProp fluid name (e.g. "R134a").
    T_evap_C, T_cond_C : float
        Saturation temperatures of evaporator / condenser, deg C.
    superheat_K, subcooling_K : float
        Degrees of superheat at compressor inlet and subcooling at
        condenser outlet, K (>= 0).
    eta_isentropic : float
        Compressor isentropic efficiency, 0 < eta <= 1.

    Returns
    -------
    CycleResult
    """
    if T_cond_C <= T_evap_C:
        raise ValueError("Condensing temperature must exceed evaporating "
                          "temperature.")
    if not (0 < eta_isentropic <= 1):
        raise ValueError("Isentropic efficiency must be in (0, 1].")

    T_evap_K = T_evap_C + C_TO_K
    T_cond_K = T_cond_C + C_TO_K

    P_evap = PropsSI("P", "T", T_evap_K, "Q", 1, fluid)
    P_cond = PropsSI("P", "T", T_cond_K, "Q", 0, fluid)

    # State 1: compressor inlet (superheated vapor at P_evap)
    T1_K = T_evap_K + superheat_K
    h1 = PropsSI("H", "T", T1_K, "P", P_evap, fluid)
    s1 = PropsSI("S", "T", T1_K, "P", P_evap, fluid)

    # State 2s: isentropic compression to P_cond
    h2s = PropsSI("H", "P", P_cond, "S", s1, fluid)
    T2s_K = PropsSI("T", "P", P_cond, "H", h2s, fluid)

    # State 2: actual compression accounting for isentropic efficiency
    h2 = h1 + (h2s - h1) / eta_isentropic
    T2_K = PropsSI("T", "P", P_cond, "H", h2, fluid)

    # State 3: condenser outlet (subcooled liquid at P_cond)
    T3_K = T_cond_K - subcooling_K
    h3 = PropsSI("H", "T", T3_K, "P", P_cond, fluid)

    # State 4: evaporator inlet (isenthalpic expansion to P_evap)
    h4 = h3
    T4_K = PropsSI("T", "P", P_evap, "H", h4, fluid)

    q_evap = h1 - h4
    q_cond = h2 - h3
    w_comp = h2 - h1

    cop_heating = q_cond / w_comp
    cop_cooling = q_evap / w_comp

    return CycleResult(
        fluid=fluid, T_evap_C=T_evap_C, T_cond_C=T_cond_C,
        superheat_K=superheat_K, subcooling_K=subcooling_K,
        eta_isentropic=eta_isentropic,
        P_evap_Pa=P_evap, P_cond_Pa=P_cond,
        T1_C=T1_K - C_TO_K, T2_C=T2_K - C_TO_K, T2s_C=T2s_K - C_TO_K,
        T3_C=T3_K - C_TO_K, T4_C=T4_K - C_TO_K,
        h1=h1, h2=h2, h2s=h2s, h3=h3, h4=h4,
        q_evap=q_evap, q_cond=q_cond, w_comp=w_comp,
        cop_heating=cop_heating, cop_cooling=cop_cooling,
        pressure_ratio=P_cond / P_evap,
    )


def optimize_cycle(fluid, T_source_C, T_sink_C, superheat_K, subcooling_K,
                    eta_isentropic, min_approach_K, max_approach_K,
                    T_max_discharge_C, mode="heating", n_grid=5):
    """
    Find the evaporating/condensing saturation temperatures (within
    allowed heat-exchanger approach limits) that maximize COP, subject
    to a maximum compressor discharge temperature.

    T_evap is searched in [T_source - max_approach, T_source - min_approach]
    T_cond is searched in [T_sink + min_approach, T_sink + max_approach]

    A small multi-start SLSQP search is used (SLSQP can stall from a
    single poor starting point on this near-linear objective), and the
    best feasible point found is returned together with the full cycle
    solution and the optimizer's convergence info.
    """
    bounds = [
        (T_source_C - max_approach_K, T_source_C - min_approach_K),
        (T_sink_C + min_approach_K, T_sink_C + max_approach_K),
    ]

    def cop_of(x):
        T_evap_C, T_cond_C = x
        try:
            res = solve_cycle(fluid, T_evap_C, T_cond_C, superheat_K,
                               subcooling_K, eta_isentropic)
        except Exception:
            return None
        return res

    def objective(x):
        res = cop_of(x)
        if res is None:
            return 1e3
        cop = res.cop_heating if mode == "heating" else res.cop_cooling
        return -cop

    def discharge_constraint(x):
        res = cop_of(x)
        if res is None:
            return -1e3
        return T_max_discharge_C - res.T2_C  # must be >= 0

    constraints = [{"type": "ineq", "fun": discharge_constraint}]

    starts_e = np.linspace(bounds[0][0], bounds[0][1], n_grid)
    starts_c = np.linspace(bounds[1][0], bounds[1][1], n_grid)

    best = None
    for te0 in starts_e:
        for tc0 in starts_c:
            if tc0 - te0 < 1.0:
                continue
            try:
                r = minimize(objective, [te0, tc0], method="SLSQP",
                             bounds=bounds, constraints=constraints,
                             options={"maxiter": 200, "ftol": 1e-9})
            except Exception:
                continue
            feasible = discharge_constraint(r.x) >= -1e-3
            if not feasible:
                continue
            if best is None or r.fun < best.fun:
                best = r

    if best is None:
        raise RuntimeError(
            "No feasible operating point found - try relaxing the "
            "discharge-temperature limit or the approach temperatures."
        )

    best_state = cop_of(best.x)
    return best_state, best
