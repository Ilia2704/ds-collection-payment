# ds-collection-payment

## Overview

End-to-end credit collections analytics pipeline covering:

- payment event prediction (classification)
- expected payment amount modeling (regression)
- portfolio segmentation
- expected value–based decisioning

Designed as a production-style workflow similar to real microfinance / lending risk systems.

Demonstrates how data science supports operational collections strategy and cash flow forecasting.

Production-style end-to-end project for **collection payment prediction** and **payment amount modeling** using business data.

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

#### Results 

<img width="576" height="427" alt="image" src="https://github.com/user-attachments/assets/b6a55661-9046-4cfe-8413-669a122896b8" />

<img width="797" height="496" alt="image" src="https://github.com/user-attachments/assets/6ca8e27b-db9e-47cb-b865-0024cfa12061" />

<img width="533" height="406" alt="image" src="https://github.com/user-attachments/assets/177e03e9-1492-444c-83e8-ff02ff2ecac6" />

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

#### Results

<img width="852" height="411" alt="image" src="https://github.com/user-attachments/assets/1cf69e2f-71e3-4329-841b-eb31974b3fb7" />

<img width="853" height="306" alt="image" src="https://github.com/user-attachments/assets/6ddcebc3-73e9-4d33-a4fb-1d71a0536128" />

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

## Key concepts 

* Data generation with realistic distributions
* Overdue bucket logic (`5–30`, `30–60`, `60–90`, `90–180`, `180–360`)
* Product-level portfolio segmentation
* Custom feature transformers
* Clear separation between:

  * notebooks (experiments and storytelling)
  * python modules (`transformers/`)

Note
Uses synthetic but statistically realistic data to preserve confidentiality.
Modeling logic mirrors real-world collection risk pipelines.


