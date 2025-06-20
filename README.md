````markdown
# Fintech Credit-Scoring Seminar  
Synthetic Data & Model Benchmarks

A minimal, reproducible pipeline that

* builds six synthetic datasets (scorable / unscorable, three generators),
* trains logistic-regression and random-forest baselines,
* outputs fidelity checks and ROC figures.

---

## Quick Start

```bash
git clone https://github.com/<user>/fintech-synthetic-credit.git
cd fintech-synthetic-credit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# generate data
python notebooks/02a_synthetic_data_generation_scorable_population.py
python notebooks/02b_synthetic_data_generation_unscorable_population.py

# summarise + train
python notebooks/03_data_summarizer.py
python notebooks/08_balanced_modeling.py
````

Figures appear in `figures/`, results in `data/`.

---

## Key Scripts

| Script                          | Role                                |
| ------------------------------- | ----------------------------------- |
| `02a_*_scorable_*.py`           | Scorable synthetic datasets         |
| `02b_*_unscorable_*.py`         | Unscorable synthetic datasets       |
| `03_data_summarizer.py`         | Marginal-gap and KS checks          |
| `08_balanced_modeling.py`       | Logit and RF with balanced training |
| `create_publication_visuals.py` | Marginal bars, heat maps, ROC       |

---

## Dependencies

pandas · numpy · scikit-learn · sdv · matplotlib

Full list in `requirements.txt`.

---

## AI Note

Code scaffolding was drafted with Claude 4 and GPT-4; all scripts were executed, debugged, and verified by the author before release.

```
```
