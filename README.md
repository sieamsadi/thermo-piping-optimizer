# Thermodynamic Cycle & Piping Network Optimizer

A portfolio-ready engineering tool for mechanical engineering freelance work,
combining open-source fluid properties (**CoolProp**) with numerical
optimization (**scipy.optimize**) in an interactive **Streamlit** app.

## What it does

1. **Heat Pump / Refrigeration Cycle Optimizer** (`cycle_model.py`)
   - Models a single-stage vapor-compression cycle for ~9 common
     refrigerants (R134a, R410A, R32, R290, Ammonia, CO2, etc.)
   - Uses a constrained, multi-start SLSQP search
     (`scipy.optimize.minimize`) to find the evaporating/condensing
     saturation temperatures that **maximize COP**, subject to
     heat-exchanger approach-temperature limits and a compressor
     discharge-temperature ceiling.
   - Plots a COP sensitivity heatmap so a client can see the trade-off
     space, not just a single number.

2. **Piping Network Sizing** (`piping_model.py`)
   - Pressure-drop calculator using the Darcy-Weisbach equation with
     the **Colebrook-White** friction factor solved exactly via
     `scipy.optimize.brentq` (root-finding on the implicit equation,
     not the Swamee-Jain approximation).
   - **Economic pipe diameter** optimizer (`scipy.optimize.minimize_scalar`):
     minimizes annualized capital cost + pumping energy cost, the
     classic chemical/mechanical engineering pipe-sizing trade-off.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

This was developed in a sandbox without internet access, so the
CoolProp/Streamlit/Plotly install and a live run could not be verified
end-to-end here — please run it locally first and let me know if you hit
any errors (most likely culprits would be a CoolProp fluid-name typo or
a Streamlit version mismatch) and I can fix them immediately.

## Suggested portfolio framing

- Screenshot the COP heatmap and the cost-vs-diameter chart for your
  portfolio site/LinkedIn — these communicate "I understand the
  underlying trade-off," which is what clients are actually paying for.
- Mention specifically: *open-source REFPROP alternative* (CoolProp) +
  *numerical optimization* (scipy) — these are the two keywords likely
  to catch the attention of HVAC/process engineering firms scanning for
  freelancers who can do more than spreadsheet calculations.
- The "About" tab in the app already lists honest limitations and
  natural next steps (branched networks, multi-stage cycles, PDF
  reports) — use that list as a menu of paid add-on scopes when you
  pitch follow-on work to a client.

## File structure

```
thermo_piping_optimizer/
├── app.py            # Streamlit UI (3 tabs: cycle, piping, about)
├── cycle_model.py     # Vapor-compression cycle + COP optimizer
├── piping_model.py    # Pressure drop + economic diameter optimizer
├── requirements.txt
└── README.md
```
