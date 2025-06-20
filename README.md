```markdown
# Fintech Credit-Scoring Seminar

A reproducible pipeline that  

1. builds six synthetic credit-risk datasets (scorable / unscorable; basic, copula, CTGAN),  
2. trains logistic-regression and random-forest models with balanced training,  
3. reports fidelity checks and ROC-based performance figures.

---

## Structure
```

notebooks/                    ← main Python scripts
├─ synthetic\_data\_generation\_scorable\_population.py
├─ synthetic\_data\_generation\_unscorable\_population.py
├─ balanced\_modeling.py
├─ full\_test\_modeling.py
├─ marginal\_fidelity\_assessment.py
└─ create\_publication\_visuals.py
data/                         ← generated CSV files
figures/                      ← PDF figures
requirements.txt              ← pinned packages

````

---

## Quick start
```bash
git clone https://github.com/<user>/fintech-synthetic-credit.git
cd fintech-synthetic-credit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# generate datasets
python notebooks/synthetic_data_generation_scorable_population.py
python notebooks/synthetic_data_generation_unscorable_population.py

# run quality checks and models
python notebooks/marginal_fidelity_assessment.py
python notebooks/balanced_modeling.py

# create publication figures
python notebooks/create_publication_visuals.py
````

Outputs land in `data/` and `figures/`.

---

## Key components

| Script                            | Purpose                                         |
| --------------------------------- | ----------------------------------------------- |
| `*_scorable_population.py`        | synthetic data for borrowers with bureau scores |
| `*_unscorable_population.py`      | synthetic data for thin-file borrowers          |
| `balanced_modeling.py`            | logit + random forest with rare-event sampling  |
| `full_test_modeling.py`           | balanced train plus independent hold-out test   |
| `marginal_fidelity_assessment.py` | χ², KS, and gap statistics                      |
| `create_publication_visuals.py`   | bar charts, heat maps, ROC curves               |

Core libs: pandas · numpy · scikit-learn · sdv (CTGAN) · matplotlib.

AI note: code scaffolding drafted with Claude 4 and GPT-4; all scripts were executed and checked by the author.

```
```
