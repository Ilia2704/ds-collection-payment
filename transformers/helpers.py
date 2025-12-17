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


"""
add_ww_mm_date
df_filter_by_dts
calculate_information_values
get_num_feature_stability_index
get_cat_feature_stability_index
"""

def add_ww_mm_date(df, ts_col):
    df['dt_week'] = df[ts_col].apply(lambda dt: (dt - timedelta(days=dt.weekday())).date())
    df['dt_month'] = df[ts_col].apply(lambda dt: dt.replace(day=1).date())
    return df


def df_filter_by_dts(df, dt_start, dt_end, date_col):
    return df[(df[date_col] >= dt_start) & (df[date_col] < dt_end)].copy()


def calculate_information_values(df, feature_columns, target_column):
    iv_values = {}

    for feature in feature_columns:
        # Calculate IV for the feature
        positive_counts = df[df[target_column] == 1][feature].value_counts()
        negative_counts = df[df[target_column] == 0][feature].value_counts()

        positive_total = positive_counts.sum()
        negative_total = negative_counts.sum()

        positive_event_rate = positive_counts / positive_total
        negative_event_rate = negative_counts / negative_total

        iv = np.sum((positive_event_rate - negative_event_rate) * np.log(positive_event_rate / negative_event_rate))

        iv_values[feature] = iv

        df_iv = pd.DataFrame.from_dict(iv_values, orient='index')
        df_iv = df_iv.reset_index().rename(columns={'index': 'Feature', 0: 'IV'})
        df_iv = df_iv.sort_values(by='IV', ascending=False)
        df_iv = df_iv.reset_index(drop=True)

    return df_iv


def get_num_feature_stability_index(df, month_col, feature_col, target_col):
    """
    Calculate the weighted stability score for a feature's correlation with the target over time,
    considering both the mean absolute correlation, correlation variance, and sample sizes.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - month_col (str): Column name for the time periods (e.g., 'dt_month').
    - feature_col (str): Column name for the numerical feature.
    - target_col (str): Column name for the binary target.

    Returns:
    - float: Weighted stability score for the feature.
    """
    # Ensure the month column is datetime
    df = df.copy()
    df[month_col] = pd.to_datetime(df[month_col], format='%Y-%m-%d')

    # Exclude grouping columns explicitly to avoid DeprecationWarning
    def calculate_stats(group):
        # Ensure there are enough data points and non-constant columns
        if len(group) > 1 and group[feature_col].std() > 0 and group[target_col].std() > 0:
            correlation = np.corrcoef(group[feature_col], group[target_col])[0, 1]
        else:
            correlation = np.nan
        return pd.Series({
            'correlation': correlation,
            'sample_size': len(group)
        })

    # Group by month and calculate correlation and sample size manually
    monthly_stats = df.groupby(month_col)[[feature_col, target_col]].apply(calculate_stats).reset_index()

    # Drop rows where correlation could not be computed
    monthly_stats = monthly_stats.dropna(subset=['correlation'])

    # If no valid correlations exist, return NaN
    if monthly_stats.empty:
        return np.nan

    # Calculate the weighted mean absolute correlation
    weighted_abs_corr = np.average(monthly_stats['correlation'].abs(), weights=monthly_stats['sample_size'])

    # Calculate the weighted variance of correlation
    weighted_corr_variance = np.average((monthly_stats['correlation'] - weighted_abs_corr)**2, weights=monthly_stats['sample_size'])

    # Calculate the weighted stability score
    stability_score = weighted_abs_corr / (1 + weighted_corr_variance)

    return stability_score


def get_cat_feature_stability_index(df, month_col, feature_col, target_col):
    """
    Calculate stability of a categorical feature based on relative differences in mean target values,
    considering only consecutive categories based on overall mean target.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - month_col (str): Column name for the time periods (e.g., 'dt_month').
    - feature_col (str): Column name for the categorical feature.
    - target_col (str): Column name for the target.

    Returns:
    - float: Stability score for the feature (normalized between 0 and 1).
    """
    
    # Ensure the month column is datetime
    df = df.copy()
    df[month_col] = pd.to_datetime(df[month_col], format='%Y-%m-%d')

    # Step 1: Compute overall mean target for each category (global mean)
    overall_means = df.groupby(feature_col)[target_col].mean().sort_values().reset_index()
    overall_means['rank'] = range(len(overall_means))  # Assign ranks

    # Step 2: Compute mean target and sample size per month and category
    grouped = df.groupby([month_col, feature_col]).agg(
        mean_target=(target_col, 'mean'),
        sample_size=(target_col, 'size')
    ).reset_index()

    # Step 3: Pivot to get mean target and sample size by category over months
    pivot_mean = grouped.pivot(index=month_col, columns=feature_col, values='mean_target').fillna(0)
    pivot_count = grouped.pivot(index=month_col, columns=feature_col, values='sample_size').fillna(0)

    # Step 4: Normalize mean targets within each month
    pivot_mean_normalized = pivot_mean.div(pivot_mean.sum(axis=1), axis=0)

    # Step 5: Calculate pairwise differences for consecutive categories
    pairwise_variances = []
    for i in range(len(overall_means) - 1):
        cat1 = overall_means.iloc[i][feature_col]
        cat2 = overall_means.iloc[i + 1][feature_col]

        # Calculate relative difference
        relative_diff = pivot_mean_normalized[cat1] - pivot_mean_normalized[cat2]

        # Compute variance of the relative differences over months
        variance = relative_diff.var()
        pairwise_variances.append(variance)

    # Step 6: Aggregate variances into a single stability score
    # Normalize variances to the range [0, 1]
    max_variance = max(pairwise_variances) if pairwise_variances else 1e-6
    normalized_variances = [v / max_variance for v in pairwise_variances]

    # Compute weighted variance
    total_sample_weights = pivot_count.sum().sum()
    normalized_weights = pivot_count.sum(axis=1) / total_sample_weights
    weighted_variance = sum(variance * weight for variance, weight in zip(normalized_variances, normalized_weights))

    # Step 7: Calculate stability score as a penalty of variance
    stability_score = 1 - weighted_variance  # High variance leads to lower scores, bounded between 0 and 1.

    return max(0, min(1, stability_score)) # Ensure score stays in [0, 1]
