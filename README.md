# ds-collection-payment

End-to-end demo project for **collection payment prediction** and **payment amount modeling** using synthetic but business-realistic data.

The repository demonstrates:

* feature engineering for collections / arrears use cases
* binary classification (payment event)
* regression (payment amount)
* portfolio segmentation by product and overdue buckets
* clean, reproducible ML experimentation in Jupyter


---

## Project structure

```text
ds-collection-payment/
├── README.md
├── 01_collection_payment_prediction.ipynb
├── 02_collection_payment_amount.ipynb
├── requirements.txt
└── transformers/
    ├── __init__.py
    ├── simple_features_transformer.py
    ├── bin_funcs.py
    ├── segmentation.py
    ├── preparing.py
    ├── quant_classes.py
    ├── py_classes.py
    ├── helpers.py
    ├── constants.py
    └── logger.py
```

## Notebooks overview

### `01_collection_payment_prediction.ipynb`

#### Goal

Build a **binary classification model** to predict whether a customer will make a payment during the collection window.

**Business motivation:**

* prioritize collection efforts
* reduce operational cost
* improve contact strategy by focusing on high-probability payers

#### Target

* `target = 1` → payment occurred
* `target = 0` → no payment

#### Key steps

* synthetic but realistic data generation
* feature engineering (behavioral, arrears, bureau-style features)
* product and overdue-bucket segmentation
* model training (Logistic Regression / CatBoost)
* evaluation with risk-focused metrics

#### Metrics

* ROC AUC
* Gini coefficient
* KS statistic
* Precision / Recall
* Distribution of predictions by overdue bucket

#### Results (demo-level)

* Stable ROC AUC and Gini across main product segments
* Clear monotonic degradation of payment probability with higher overdue buckets
* Predictive lift concentrated in early buckets (5–30, 30–60), aligning with collection intuition

---

### `02_collection_payment_amount.ipynb`

#### Goal

Estimate the **expected payment amount**, conditional on a payment occurring.

**Business motivation:**

* optimize promise-to-pay strategies
* forecast cash inflow from collections
* support expected value–based decisioning

#### Target

* `amount_of_payment`
* modeled only for observations where `target = 1`

#### Key steps

* filtering to paid cases
* regression modeling
* feature reuse from classification pipeline
* evaluation by product and overdue bucket

#### Metrics

* MAE (Mean Absolute Error)
* RMSE
* Error distribution by overdue bucket
* Comparison of predicted vs actual payment amounts

#### Results (demo-level)

* Reasonable error stability across product segments
* Higher variance in late overdue buckets (expected behavior)
* Stronger signal in promised amount and recent payment behavior features

---

## Environment setup (recommended)

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install \
  --user \
  --name ds-collection-payment \
  --display-name "Python (ds-collection-payment)"
```

---

## Key concepts demonstrated

* Synthetic data generation with realistic distributions
* Overdue bucket logic (`5–30`, `30–60`, `60–90`, `90–180`, `180–360`)
* Product-level portfolio segmentation
* Custom feature transformers
* Clear separation between:

  * notebooks (experiments and storytelling)
  * python modules (`transformers/`)



## Disclaimer
This project is intended **for demonstration and educational purposes only**.


