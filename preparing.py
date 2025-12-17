# general imports

import logging
import json
import gc
import os
import pickle
import joblib
import re
import math
import warnings
import string
from datetime import timedelta, datetime
from itertools import combinations, groupby

import numpy as np
import pandas as pd
import boto3
from io import BytesIO
from pandas.errors import PerformanceWarning

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

from feature_engine.encoding.rare_label import RareLabelEncoder
from feature_engine.encoding import WoEEncoder
from feature_engine.selection import (
    DropDuplicateFeatures,
    DropConstantFeatures,
    DropCorrelatedFeatures,
    SelectByTargetMeanPerformance
)

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    auc,
    log_loss,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    train_test_split,
)

from optbinning import OptimalBinning

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from transformers.constants import (
    DT_LABEL,
    TS_COL,
    TARGET,
    MISSING,
    OTHER,
    CAT,
    NUM,
    BIN,
    WOE,
    SEP,
    DT_START,
    DT_END,
)

from transformers.logger import logger


'''
read_data,
drop_missing_json_rows,
parse_json_field,
filter_columns,
compute_salary_median,
merge_cb_columns,
merge_gender,
merge_dob,
merge_labels,
parse_dates,
calculate_age_experience
'''


def read_data(path) -> pd.DataFrame:
    logger.info(f"Reading data from: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns.")
    return df
    

def compute_salary_median(df: pd.DataFrame) -> pd.DataFrame:
    salary_columns = [
        'salary1', 
        'salary2', 
        'salary3',
        'salary4', 
        'salary5', 
        'salary6',
        'mbl_income',
        'final_income',
        'salary_amount1', 
        'salary_amount2', 
        'salary_amount3',
        'salary_amount4', 
        'salary_amount5', 
        'salary_amount6'
    ]
    
    df[salary_columns] = df[salary_columns].apply(pd.to_numeric, errors='coerce')
    df['salary_median'] = df[salary_columns].median(axis=1, skipna=True).fillna(0.0)
    df.drop(columns=salary_columns, inplace=True)
    logger.info(f"Computed 'salary_median' and dropped {len(salary_columns)} salary columns.")
    return df

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df['extractdate'] = pd.to_datetime(df['extractdate'], errors='coerce', utc=True).dt.tz_localize(None)
    df['date_joined_company'] = pd.to_datetime(df['date_joined_company'],
                                                               errors='coerce',
                                                               utc=True).dt.tz_localize(None)
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce', utc=True).dt.tz_localize(None)
    logger.info("Parsed 'extractdate', 'date_joined_company', and 'date_of_birth' to datetime.")
    return df

def calculate_age_experience(df: pd.DataFrame) -> pd.DataFrame:
    df['age'] = ((df['extractdate'] - df['date_of_birth']).dt.days / 365).round().astype('Int64')
    df['experience'] = ((df['extractdate'] - df['date_joined_company']).dt.days / 365).round().astype('Int64')

    df['age'] = df['age'].where(df['date_of_birth'].notna(), pd.NA)
    df['experience'] = df['experience'].where(df['date_joined_company'].notna(), pd.NA)

    df.drop(columns=['date_joined_company', 
                     'date_of_birth'
                    ], 
            inplace=True)
    logger.info("Calculated 'age' and 'experience', and dropped temporary columns.")
    return df


def rename_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Renaming features...")

    rename_dict = {
    "effective_interest_rate": "application_effectiveinterestrate",
    "state_of_origin": "bvn_stateoforigin",
    "label": "labeling",
    "employment_sector": "application_employmentsector",
    "employment_status": "application_employmentstatus",
    "existing_customer": "application_existingcustomer",
    "last_loan_status": "application_lastloanstatus",
    "loan_type": "application_loantype",
    "employer_lga": "customer_employment_employerlga",
    "business_sector": "customer_employment_businesssector",
    "address_lga": "customer_addresslga",
    "cb_monthly_inst": "cb_avg_totalmonthlyinstallments",
    "cb_outstanding": "cb_avg_outstandingloan",
    "cb_no_loans": "cb_avg_noofloans",
    "cb_max_dpd": "cb_avg_maxdpd",
    "cb_repayment_status": "cb_avg_highestrepaymentstatus",
    "cb_highest_repayment": "cb_avg_highestloanrepayment",
    "cb_current_dpd": "cb_avg_currentdpd",
    "requested_loan_amount": "application_requestedloanamount"
    }

    df_renamed = df.rename(columns=rename_dict)
    logger.info("Features were renamed.")

    return df_renamed


