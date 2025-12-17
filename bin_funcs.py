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

# import from transformers

from transformers.constants import (
    DT_LABEL,
    TS_COL,
    TARGET,
    MISSING,
    NAN,
    OTHER,
    CAT,
    NUM,
    BIN,
    WOE,
    SEP,
    GROUPS,
    DT_START,
    DT_END,
)

from transformers.logger import logger

from transformers.preparing import (
    read_data,
    compute_salary_median,
    parse_dates,
    calculate_age_experience
)


def get_optbin_info_num_collection(data, 
                                   feature, 
                                   target=TARGET, 
                                   max_n_bins=4, 
                                   min_bin_size=0.09, 
                                   min_target_diff=0.02):
    
    x = pd.to_numeric(data[feature], errors='coerce').astype(float).fillna(NAN)
    y = data[target].values

    optb = OptimalBinning(    
        dtype="numerical", 
        solver="cp", 
        prebinning_method="cart",
        min_event_rate_diff=min_target_diff, # minimal difference in event rate
        divergence='iv',                     # objective metric to maximize
        min_bin_size=min_bin_size,           # minimal fraction for bin size
        # max_pvalue=0.05,                   # maximum p-value among bins - turned off as affects too much
        time_limit=10,
        min_prebin_size=0.01,
        max_n_prebins=50,
        max_n_bins=max_n_bins
    )
    
    optb.fit(x, y)
    bins_lst = [x.min()] + list(optb.splits) + [np.inf]

    return bins_lst


def get_optbin_info_cat_collection(data, 
                                   feature, 
                                   target=TARGET, 
                                   max_n_bins=4, 
                                   min_bin_size=0.10, 
                                   min_target_diff=0.02):

    x = data[feature].fillna(MISSING).values.astype(str)
    y = data[target].values

    optb = OptimalBinning(
        dtype="categorical",
        solver="mip",
        prebinning_method="cart",
        min_event_rate_diff=min_target_diff,  # minimal difference in event rate
        divergence='iv',                      # objective metric to maximize
        min_bin_size=min_bin_size,            # minimal fraction for bin size
        # max_pvalue=0.05,                    # maximum p-value among bins - turned off as affects too much
        max_n_bins=max_n_bins,
        time_limit=10,
        min_prebin_size=0.01,
        max_n_prebins=50,
        monotonic_trend=None  # ONLY FOR COLLECTION
    )

    optb.fit(x, y)
    splits_array = optb.splits

    if not any(OTHER in t for t in splits_array):
        splits_array.append(np.array([OTHER], dtype=object))

    groups_map_dct = {frozenset(split): GROUPS[i] for i, split in enumerate(splits_array)}

    return groups_map_dct

def get_values_map(input_map):
    output_map = {}

    for key_set, value in input_map.items():
        for v in key_set:
            output_map[v] = value

    return output_map