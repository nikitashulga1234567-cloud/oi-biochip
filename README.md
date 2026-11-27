# OI Biochip Simulator v4

Experimental Python simulator inspired by organoid intelligence (OI) and high-density microelectrode arrays (HD-MEA). This is a computational research prototype, not a physical biochip or a biologically validated digital twin.

## Features
- Conductance-based RS/FS neurons (Pospischil et al., 2008), AMPA/NMDA/GABA-A synapses.
- 3D network, simulated gel environment and 20 × 20 electrode array.
- Reward-modulated STDP, eligibility traces, reward critic and homeostatic scaling.
- A/B stimulus discrimination, mutual information and transfer entropy.
- Legacy v3 Pong simulator, separate from the v4 A/B engine.

## Run
Use Python 3.10+ (recommended) in a virtual environment:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python oi_v4_checks.py fi
python oi_v4_checks.py ab --seeds 1 --n 1000 --train 900 --eval 200 --procs 1
```

Full experiments may take substantial time. The original notebook can also be opened in Google Colab; its first code cell reconstructs the package. The optional Google Sheets cell requests Google authentication and is not required for simulation.

## Research status and limitations
The accompanying research report (not bundled here) describes later v4.1 experiments, including structural plasticity, which is not implemented in this v4 snapshot. Reported A/B accuracy was 65.4% on seeds 1–4; a later series reported 78.5% without and 84.1% with structural plasticity on seeds 5–8. The difference was not statistically significant (p=0.375). These are historical report values, not reproduced by this release. Transfer entropy was not significant. Dense N×N arrays limit scalability. Pong results belong to the legacy v3 engine.

## References
- Pospischil et al. (2008), Minimal Hodgkin–Huxley type models, https://doi.org/10.1007/s00422-008-0263-8
- Hodgkin & Huxley (1952), https://doi.org/10.1113/jphysiol.1952.sp004764
- Izhikevich (2007), Solving the distal reward problem through linkage of STDP and dopamine.

## License
No license has been selected. The author should choose a license before inviting reuse.
