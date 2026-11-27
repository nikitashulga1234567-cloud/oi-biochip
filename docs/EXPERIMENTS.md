# Experiment notes and interpretation

This document distinguishes reported historical results from checks runnable using the source currently published in this repository.

## Versions and scope

- The **v4 source snapshot** contains the conductance-based 3D `BioNet`, `HDMEA`, `RSTDP` and A/B classification task.
- The accompanying **research report includes later v4.1 experiments** with structural plasticity. That later implementation is **not fully present** in this v4 source snapshot.
- The **v3 Pong engine** is a separate implementation. Its results do not demonstrate Pong learning by the v4 biophysical network.

## Reported A/B classification results

| Source-described condition | Seeds | Held-out accuracy |
| --- | --- | ---: |
| Early A/B series | 1–4 | 65.4% |
| Later baseline (without structural plasticity) | 5–8 | 78.5% |
| Later structural-plasticity condition | 5–8 | 84.1% |

The later baseline-to-structural difference is **5.6 percentage points**. The report describes it as **not statistically significant** (`p = 0.375`). There are too few experimental seeds to claim a robust effect. Different seed groups should not be merged into one controlled comparison.

Additional report values include stimulus–response mutual information of **0.219 bits** and a non-significant sensory-to-motor transfer-entropy result (`p = 0.94`). These are reported measurements, not independently recomputed results for this published snapshot.

The graph in `assets/ab-results.svg` faithfully represents the reported group means as an **illustrative comparison**, without confidence intervals, because the underlying per-seed data were not published in this repository.

## Running checks from this repository

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run quick smoke tests:

```bash
python -m unittest discover -s tests -v
```

Check single-neuron dynamics / f–I curves:

```bash
python oi_v4_checks.py fi
```

Run a v4 A/B experiment:

```bash
python oi_v4_checks.py ab --seeds 1 --n 1000 --train 900 --eval 200 --procs 1
```

A/B runs can take substantial compute. Save the full configuration, random seeds, train/evaluation split and package versions when reporting new results. Do **not** treat these commands as reproducing the v4.1 structural-plasticity findings.

## Proposed scientific follow-up

- Publish the v4.1 structural-plasticity source and experiment configs, if available.
- Report per-seed scores and uncertainty intervals using a larger preregistered seed set.
- Separate model selection seeds from final held-out test seeds.
- Validate firing dynamics against electrophysiology and relevant reference measurements.
- Replace dense all-pairs arrays with sparse network structures before larger-scale experiments.

**Source:** [OI Biochip research report (PDF)](OI_biochip.pdf).
