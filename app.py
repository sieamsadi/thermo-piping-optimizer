"""
app.py
-------
Thermodynamic Cycle & Piping Network Optimizer
A portfolio project for mechanical engineering freelancing:
  Tab 1: Vapor-compression heat pump / refrigeration cycle COP optimizer
  Tab 2: Piping network pressure-drop calculator & economic pipe-size optimizer

Built with CoolProp (open-source REFPROP alternative) + scipy.optimize.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cycle_model import REFRIGERANTS, solve_cycle, optimize_cycle
from piping_model import (
    EconomicDiameterInputs, annualized_cost, optimize_economic_diameter,
    evaluate_network, PipeSegment,
)

st.set_page_config(
    page_title="Thermo-Fluid Optimizer",
    page_icon="🌀",
    layout="wide",
)

st.title("🌀 Thermodynamic Cycle & Piping Network Optimizer")
st.caption(
    "Open-source fluid properties (CoolProp) + numerical optimization "
    "(scipy.optimize) for HVAC and process piping design work."
)

tab_cycle, tab_piping, tab_about = st.tabs(
    ["❄️ Heat Pump / Refrigeration Cycle", "🔧 Piping Network Sizing", "ℹ️ About"]
)

# ----------------------------------------------------------------------
# TAB 1 — Vapor compression cycle COP optimizer
# ----------------------------------------------------------------------
with tab_cycle:
    st.subheader("Vapor-Compression Cycle: COP Optimizer")
    st.write(
        "Given source/sink temperatures (e.g. outdoor air and a heating "
        "loop, or a process stream to be cooled), find the evaporating "
        "and condensing saturation temperatures that **maximize COP**, "
        "subject to realistic heat-exchanger approach limits and a "
        "compressor discharge-temperature ceiling."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        fluid = st.selectbox("Refrigerant", REFRIGERANTS, index=0)
        mode = st.radio("Optimize for", ["heating", "cooling"], horizontal=True)
    with col2:
        T_source_C = st.number_input("Source temperature (°C)", value=2.0,
                                      help="e.g. outdoor air temp for an "
                                           "air-source heat pump evaporator, "
                                           "or process fluid being cooled.")
        T_sink_C = st.number_input("Sink temperature (°C)", value=45.0,
                                    help="e.g. heating-loop supply temp, "
                                         "or cooling-water temp for a "
                                         "condenser.")
    with col3:
        superheat_K = st.number_input("Superheat (K)", value=5.0, min_value=0.0)
        subcooling_K = st.number_input("Subcooling (K)", value=5.0, min_value=0.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        eta_isentropic = st.slider("Compressor isentropic efficiency", 0.4, 1.0, 0.75)
    with col5:
        min_approach_K = st.number_input("Min. HX approach (K)", value=5.0, min_value=0.1)
        max_approach_K = st.number_input("Max. HX approach (K)", value=15.0, min_value=0.2)
    with col6:
        T_max_discharge_C = st.number_input("Max compressor discharge T (°C)",
                                              value=120.0)

    if max_approach_K <= min_approach_K:
        st.error("Max approach must be greater than min approach.")
    else:
        if st.button("🔍 Optimize cycle", type="primary"):
            try:
                with st.spinner("Running constrained optimization (scipy SLSQP, multi-start)..."):
                    best_state, opt_info = optimize_cycle(
                        fluid, T_source_C, T_sink_C, superheat_K, subcooling_K,
                        eta_isentropic, min_approach_K, max_approach_K,
                        T_max_discharge_C, mode=mode,
                    )
                st.success("Optimization converged.")

                cop_val = (best_state.cop_heating if mode == "heating"
                           else best_state.cop_cooling)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Optimal T_evap", f"{best_state.T_evap_C:.1f} °C")
                m2.metric("Optimal T_cond", f"{best_state.T_cond_C:.1f} °C")
                m3.metric(f"Max COP ({mode})", f"{cop_val:.2f}")
                m4.metric("Discharge temp", f"{best_state.T2_C:.1f} °C")

                st.dataframe(
                    pd.DataFrame([best_state.to_dict()]).T.rename(
                        columns={0: "Value"}),
                    use_container_width=True,
                )

                # Sensitivity surface: COP vs T_evap, T_cond (for context)
                st.markdown("**COP sensitivity around the optimum**")
                te_range = np.linspace(
                    T_source_C - max_approach_K, T_source_C - min_approach_K, 25)
                tc_range = np.linspace(
                    T_sink_C + min_approach_K, T_sink_C + max_approach_K, 25)
                Z = np.full((len(tc_range), len(te_range)), np.nan)
                for i, tc in enumerate(tc_range):
                    for j, te in enumerate(te_range):
                        if tc - te < 1.0:
                            continue
                        try:
                            r = solve_cycle(fluid, te, tc, superheat_K,
                                             subcooling_K, eta_isentropic)
                            if r.T2_C <= T_max_discharge_C:
                                Z[i, j] = (r.cop_heating if mode == "heating"
                                           else r.cop_cooling)
                        except Exception:
                            pass

                fig = go.Figure(data=go.Heatmap(
                    z=Z, x=te_range, y=tc_range, colorscale="Viridis",
                    colorbar=dict(title="COP"),
                ))
                fig.add_trace(go.Scatter(
                    x=[best_state.T_evap_C], y=[best_state.T_cond_C],
                    mode="markers+text", text=["optimum"],
                    textposition="top center",
                    marker=dict(color="red", size=12, symbol="star"),
                ))
                fig.update_layout(
                    xaxis_title="T_evap (°C)", yaxis_title="T_cond (°C)",
                    height=450, margin=dict(t=20, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Greyed-out (blank) regions are infeasible — they "
                    "violate the compressor discharge-temperature limit."
                )

            except Exception as e:
                st.error(f"Optimization failed: {e}")

    with st.expander("Or just evaluate one cycle point manually"):
        c1, c2 = st.columns(2)
        with c1:
            T_evap_manual = st.number_input("T_evap (°C)", value=-5.0, key="man_te")
        with c2:
            T_cond_manual = st.number_input("T_cond (°C)", value=50.0, key="man_tc")
        if st.button("Evaluate point"):
            try:
                r = solve_cycle(fluid, T_evap_manual, T_cond_manual,
                                 superheat_K, subcooling_K, eta_isentropic)
                st.dataframe(pd.DataFrame([r.to_dict()]).T.rename(
                    columns={0: "Value"}), use_container_width=True)
            except Exception as e:
                st.error(str(e))

# ----------------------------------------------------------------------
# TAB 2 — Piping network pressure drop + economic diameter optimizer
# ----------------------------------------------------------------------
with tab_piping:
    st.subheader("Piping Network: Pressure Drop & Economic Diameter")
    st.write(
        "Calculate pressure drop for a pipe run using CoolProp fluid "
        "properties and the Colebrook-White friction factor, or let "
        "**scipy.optimize** find the diameter that minimizes total "
        "annualized cost — pumping energy traded off against pipe "
        "capital cost (the classic 'economic pipe diameter' problem)."
    )

    sub_calc, sub_opt = st.tabs(["📐 Pressure drop calculator", "💰 Economic diameter optimizer"])

    PIPING_FLUIDS = ["Water", "INCOMP::MEG-30%", "Air", "AMMONIA", "R134a",
                      "INCOMP::MEA-30%", "CO2", "Nitrogen"]

    with sub_calc:
        c1, c2, c3 = st.columns(3)
        with c1:
            fluid_p = st.selectbox("Fluid", PIPING_FLUIDS, index=0, key="fluid_calc")
            T_C_p = st.number_input("Fluid temperature (°C)", value=20.0, key="T_calc")
        with c2:
            P_Pa_p = st.number_input("Operating pressure (kPa, abs)", value=300.0,
                                      key="P_calc") * 1000
            flow_m3h = st.number_input("Volumetric flow (m³/h)", value=50.0, key="flow_calc")
        with c3:
            D_mm = st.number_input("Pipe internal diameter (mm)", value=100.0, key="D_calc")
            L_m = st.number_input("Pipe length (m)", value=100.0, key="L_calc")

        c4, c5, c6 = st.columns(3)
        with c4:
            rough_mm = st.number_input("Absolute roughness (mm)", value=0.045,
                                        format="%.4f", key="rough_calc",
                                        help="0.045 mm ≈ commercial steel; "
                                             "0.0015 mm ≈ drawn tubing/PVC.")
        with c5:
            elev_m = st.number_input("Elevation change, outlet - inlet (m)",
                                      value=0.0, key="elev_calc")
        with c6:
            k_fit = st.number_input("Sum of minor-loss K (fittings/valves)",
                                     value=2.0, key="k_calc")

        if st.button("Calculate pressure drop", type="primary"):
            try:
                seg = PipeSegment(name="Pipe run", length_m=L_m,
                                   diameter_m=D_mm / 1000.0,
                                   roughness_m=rough_mm / 1000.0,
                                   elevation_change_m=elev_m,
                                   k_fittings=k_fit)
                results, total_dp, rho, mu = evaluate_network(
                    fluid_p, T_C_p, P_Pa_p, flow_m3h / 3600.0, [seg])
                r0 = results[0]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Velocity", f"{r0.velocity_ms:.2f} m/s")
                m2.metric("Reynolds number", f"{r0.reynolds:,.0f}")
                m3.metric("Friction factor", f"{r0.friction_factor:.4f}")
                m4.metric("Total ΔP", f"{total_dp/1000:.2f} kPa")

                st.dataframe(pd.DataFrame([{
                    "Density (kg/m³)": round(rho, 2),
                    "Viscosity (mPa·s)": round(mu * 1000, 4),
                    "ΔP friction (kPa)": round(r0.dp_friction_pa / 1000, 3),
                    "ΔP minor losses (kPa)": round(r0.dp_minor_pa / 1000, 3),
                    "ΔP elevation (kPa)": round(r0.dp_elevation_pa / 1000, 3),
                    "ΔP total (kPa)": round(total_dp / 1000, 3),
                }]).T.rename(columns={0: "Value"}), use_container_width=True)

                if r0.velocity_ms > 3.0:
                    st.warning(
                        f"Velocity ({r0.velocity_ms:.1f} m/s) is high for "
                        "typical liquid service — consider a larger "
                        "diameter to limit erosion/noise."
                    )
            except Exception as e:
                st.error(f"Calculation failed: {e}")

    with sub_opt:
        st.write(
            "Minimizes **annualized capital cost + annual pumping energy "
            "cost** over pipe diameter for a fixed required flow rate."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            fluid_o = st.selectbox("Fluid", PIPING_FLUIDS, index=0, key="fluid_opt")
            T_C_o = st.number_input("Fluid temperature (°C)", value=20.0, key="T_opt")
            P_Pa_o = st.number_input("Operating pressure (kPa, abs)", value=300.0,
                                      key="P_opt") * 1000
        with c2:
            flow_m3h_o = st.number_input("Volumetric flow (m³/h)", value=50.0, key="flow_opt")
            L_m_o = st.number_input("Pipe length (m)", value=100.0, key="L_opt")
            rough_mm_o = st.number_input("Absolute roughness (mm)", value=0.045,
                                          format="%.4f", key="rough_opt")
        with c3:
            elev_m_o = st.number_input("Elevation change (m)", value=0.0, key="elev_opt")
            k_fit_o = st.number_input("Sum of minor-loss K", value=2.0, key="k_opt")
            pump_eff = st.slider("Pump efficiency", 0.3, 0.9, 0.65, key="eff_opt")

        st.markdown("**Economics**")
        c4, c5, c6 = st.columns(3)
        with c4:
            elec_price = st.number_input("Electricity price ($/kWh)", value=0.12, key="elec_opt")
            op_hours = st.number_input("Operating hours/year", value=8000.0, key="hours_opt")
        with c5:
            capex_coeff = st.number_input(
                "Pipe cost coefficient a  [cost = a·L·D^n, $/m]",
                value=800.0, key="capex_coeff_opt",
                help="Rough installed-cost coefficient for the pipe "
                     "material/schedule in use; calibrate to a supplier "
                     "quote for real jobs.")
            capex_exp = st.number_input("Cost exponent n", value=1.5, key="capex_exp_opt")
        with c6:
            interest = st.number_input("Annual interest rate", value=0.08, key="i_opt")
            life_years = st.number_input("Pipe economic life (years)", value=15, key="n_opt")

        crf = (interest * (1 + interest) ** life_years /
               ((1 + interest) ** life_years - 1)) if interest > 0 else 1.0 / life_years

        if st.button("💡 Find economic diameter", type="primary"):
            try:
                inp = EconomicDiameterInputs(
                    fluid=fluid_o, T_C=T_C_o, P_Pa=P_Pa_o,
                    volumetric_flow_m3s=flow_m3h_o / 3600.0, length_m=L_m_o,
                    roughness_m=rough_mm_o / 1000.0,
                    elevation_change_m=elev_m_o, k_fittings=k_fit_o,
                    pump_efficiency=pump_eff,
                    electricity_price_per_kWh=elec_price,
                    operating_hours_per_year=op_hours,
                    capital_cost_coeff=capex_coeff,
                    capital_cost_exponent=capex_exp,
                    capital_recovery_factor=crf,
                )
                with st.spinner("Running scipy.optimize.minimize_scalar..."):
                    detail, result = optimize_economic_diameter(inp)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Economic diameter", f"{detail['diameter_m']*1000:.1f} mm")
                m2.metric("Velocity at optimum", f"{detail['velocity_ms']:.2f} m/s")
                m3.metric("Pump power", f"{detail['pump_power_kW']:.2f} kW")
                m4.metric("Total annualized cost", f"${detail['total_annualized_cost']:,.0f}/yr")

                st.dataframe(pd.DataFrame([{
                    "Diameter (mm)": round(detail["diameter_m"] * 1000, 1),
                    "ΔP total (kPa)": round(detail["dp_total_pa"] / 1000, 2),
                    "Reynolds number": f"{detail['reynolds']:,.0f}",
                    "Friction factor": round(detail["friction_factor"], 4),
                    "Capital cost ($)": round(detail["capital_cost"], 0),
                    "Annualized capital ($/yr)": round(detail["annualized_capital"], 0),
                    "Annual energy cost ($/yr)": round(detail["annual_energy_cost"], 0),
                    "Total annualized cost ($/yr)": round(detail["total_annualized_cost"], 0),
                }]).T.rename(columns={0: "Value"}), use_container_width=True)

                # Cost vs diameter curve
                d_sweep = np.linspace(inp.d_min_m, min(inp.d_max_m, detail["diameter_m"]*4 + 0.02), 60)
                rows = []
                for d in d_sweep:
                    try:
                        rows.append(annualized_cost(inp, d))
                    except Exception:
                        continue
                df_sweep = pd.DataFrame(rows)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_sweep["diameter_m"]*1000,
                                          y=df_sweep["annualized_capital"],
                                          name="Annualized capital cost", mode="lines"))
                fig.add_trace(go.Scatter(x=df_sweep["diameter_m"]*1000,
                                          y=df_sweep["annual_energy_cost"],
                                          name="Annual pumping energy cost", mode="lines"))
                fig.add_trace(go.Scatter(x=df_sweep["diameter_m"]*1000,
                                          y=df_sweep["total_annualized_cost"],
                                          name="Total annualized cost", mode="lines",
                                          line=dict(width=4)))
                fig.add_vline(x=detail["diameter_m"]*1000, line_dash="dash",
                               annotation_text="optimum")
                fig.update_layout(
                    xaxis_title="Pipe diameter (mm)", yaxis_title="Cost ($/yr)",
                    height=420, margin=dict(t=20, b=10),
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Small diameters minimize capital cost but cause high "
                    "velocity and friction losses, driving pumping energy "
                    "cost up; large diameters do the opposite. The optimum "
                    "balances the two — this is the classic 'economic "
                    "pipe diameter' trade-off."
                )
            except Exception as e:
                st.error(f"Optimization failed: {e}")

# ----------------------------------------------------------------------
# TAB 3 — About
# ----------------------------------------------------------------------
with tab_about:
    st.markdown("""
### About this tool

This app demonstrates two pieces of thermal-fluid engineering analysis
commonly needed by HVAC and process-piping contractors, combined with
modern open-source scientific computing:

- **CoolProp** — an open-source, REFPROP-quality fluid property database
  (NIST-validated equations of state for refrigerants, water, glycols,
  hydrocarbons, etc.)
- **scipy.optimize** — used for:
  - constrained nonlinear optimization (`minimize`, SLSQP) to find the
    cycle operating point that maximizes COP,
  - root-finding (`brentq`) to solve the implicit Colebrook-White
    friction-factor equation exactly (no Swamee-Jain approximation),
  - bounded scalar optimization (`minimize_scalar`) for the economic
    pipe-diameter trade-off.

**Engineering notes / limitations** (be transparent with clients about these):
- The vapor-compression model is a standard single-stage cycle with
  fixed superheat/subcooling — it does not include pressure drops in
  the heat exchangers/lines, multi-stage or economizer cycles, or
  part-load/cycling effects.
- The piping calculator handles a single series flow path; branched
  networks would need a network solver (e.g. Hardy-Cross or a nodal
  mass/momentum balance) as a natural next step.
- Cost coefficients in the economic-diameter tool are placeholders —
  for a real client deliverable, calibrate them to current supplier
  quotes and the client's actual electricity tariff and discount rate.

**Possible extensions for a paid engagement:** multi-stage/economizer
cycles, branched pipe networks, transient/part-load simulation,
PDF report export, or a live REFPROP backend if the client already
licenses it (CoolProp's interface is a drop-in replacement).
    """)
