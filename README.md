# Thermodynamic Cycle & Piping Network Optimizer

An interactive engineering tool that combines open-source fluid properties
(**CoolProp**, a free REFPROP-equivalent database) with numerical
optimization (**scipy.optimize**) to solve two problems HVAC and process
piping contractors deal with regularly: sizing a heat pump/refrigeration
cycle for best efficiency, and sizing pipe for lowest total cost.

**🔗 Live demo:** [thermo-piping-optimizer.streamlit.app](https://thermo-piping-optimizer.streamlit.app/) <!-- replace with your deployed URL -->

![COP sensitivity heatmap](screenshots/cop_heatmap.png)
![Economic pipe diameter chart](screenshots/economic_diameter.png)
<!-- add screenshots/ folder with these two images, or update filenames -->

## What it does

**1. Heat Pump / Refrigeration Cycle Optimizer**
- Models a single-stage vapor-compression cycle for common refrigerants
  (R134a, R410A, R32, R290, Ammonia, CO2, and more)
- Uses constrained nonlinear optimization (`scipy.optimize.minimize`,
  multi-start SLSQP) to find the evaporating/condensing temperatures
  that **maximize COP**, subject to realistic heat-exchanger approach
  limits and a compressor discharge-temperature ceiling
- Visualizes the full COP trade-off space as a sensitivity heatmap,
  not just a single optimal number

**2. Piping Network Sizing**
- Pressure-drop calculator using the Darcy-Weisbach equation with the
  **Colebrook-White** friction factor solved exactly via root-finding
  (`scipy.optimize.brentq`) — not the Swamee-Jain approximation
- **Economic pipe diameter** optimizer (`scipy.optimize.minimize_scalar`):
  balances pumping energy cost against pipe capital cost to find the
  diameter with the lowest total annualized cost

## Tech stack

`Python` · `Streamlit` · `CoolProp` · `scipy` · `NumPy` · `pandas` · `Plotly`

## Running locally

```bash
git clone https://github.com/sieamsadi/thermo-piping-optimizer.git
cd thermo-piping-optimizer
pip install -r requirements.txt
streamlit run app.py
```

## Engineering notes / limitations

- The cycle model is a standard single-stage vapor-compression cycle
  with fixed superheat/subcooling — it does not include line/HX
  pressure drops, multi-stage or economizer cycles, or part-load
  cycling effects.
- The piping calculator handles a single series flow path; branched
  networks would need a network solver (e.g. Hardy-Cross or a nodal
  balance).
- Cost coefficients in the economic-diameter tool are illustrative
  defaults — for a real project, calibrate them to current supplier
  quotes, the client's electricity tariff, and discount rate.

## File structure

```
thermo_piping_optimizer/
├── app.py             # Streamlit UI (3 tabs: cycle, piping, about)
├── cycle_model.py     # Vapor-compression cycle + COP optimizer
├── piping_model.py    # Pressure drop + economic diameter optimizer
├── requirements.txt
└── README.md
```

## About / contact

Built by [Sieam Sadi] — mechanical engineer offering thermal-fluid
analysis and engineering automation for HVAC and process design teams.
<!-- add a link to your portfolio site, LinkedIn, or email -->
