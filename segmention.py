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

from transformers.preparing import (
    read_data,
    compute_salary_median,
    parse_dates,
    calculate_age_experience
)


def split_into_segments(df, bucket_column='overdue_bucket'):
    """
    Splits the DataFrame by partial loan name match and overdue bucket.

    Returns:
        dict: {segment_bucket_name: DataFrame}
    """

    # Normalize loan_name spacing
    df = df.copy()
    df['loan_name'] = df['loan_name'].str.replace(r"\s+", " ", regex=True).str.strip()

    # Assign segment based on partial match
    def assign_segment(name):
        if pd.isna(name):
            return None
        name = name.lower()
        if '2 week' in name:
            return 'TWO_WEEK'
        if any(sub in name for sub in ['self employed', 'business loan', 'autolease', 'beta']):
            return 'SEL_BETA'
        return 'PL_ZEUS'

    df['segment'] = df['loan_name'].apply(assign_segment)

    # Normalize overdue bucket
    def normalize_bucket(bucket):
        if pd.isna(bucket):
            return None
        bucket = str(bucket).replace("–", "-").replace("_", "-").strip()
        mapping = {
            
            "5-30": "5_30",
            "30-60": "30_60",
            '180-360': "60_360",
            "60-360": "60_360",
            "30-360": "30_360",
            "60-90": "60_360",
            "360+": "60_360",
            "90-180": "60_360"  # Treat this as part of same group
        }
        return mapping.get(bucket)

    df['bucket_norm'] = df[bucket_column].apply(normalize_bucket)
    
    logger.info(f"Original DF rows: {len(df)}")
    logger.info(f"Segment distribution:\n{df['segment'].value_counts()}")
    logger.info(f"Buckets distribution:\n{df['bucket_norm'].value_counts()}")
    logger.info(f"Rows with valid bucket_norm: {df['bucket_norm'].notna().sum()}")

    # Prepare output
    results = {}

    for segment in df['segment'].dropna().unique():
        df_segment = df[df['segment'] == segment]
        if df_segment.empty:
            continue

        for bucket in df_segment['bucket_norm'].dropna().unique():
            df_sub = df_segment[df_segment['bucket_norm'] == bucket].copy()
            if not df_sub.empty:
                key = f"df_{segment}_{bucket}"
                results[key] = df_sub

    # Final logging
    total_result_rows = sum(len(d) for d in results.values())
    logger.info(f"Total rows in results: {total_result_rows}")
    logger.info(f"Lost rows (not included in results): {len(df) - total_result_rows}")

    return results


def split_into_segments_old(df, bucket_column='overdue_bucket'):
    """
    Splits the DataFrame by loan segment and overdue bucket.

    Returns:
        dict: {segment_bucket_name: DataFrame}
    """

    # Define loan segments
    categories = {
        "PL_ZEUS": [
            "INSTANT  LOAN",
            "Personal Loan - New",
            "Personal Loan - TopUp",
            "Zeus",
            "Personal Loan - Renewal",
            "Personal Loan - Buyback",
            "Personal Loan - Buyback returning"
        ],
        "TWO_WEEK": [
            "2 Week Instant Loan",
            "2 Week Zeus"
        ],
        "SEL_BETA": [
            "Self Employed - Weekly New",
            "Self Employed Renewal Weekly",
            "Self Employed New - Monthly",
            "Self Employed Loan - Weekly TopUp",
            "Self Employed Renewal Monthly",
            "Self Employed Loan - Monthly TopUp",
            "Beta Loan – NTB",
            "Beta Loan – Renewal",
            "Personal Loan – Beta NTB"
        ],
    }

    # Normalize loan_name spacing
    df = df.copy()
    df['loan_name'] = df['loan_name'].str.replace(r"\s+", " ", regex=True).str.strip()

    # Standardize overdue bucket
    def normalize_bucket(bucket):
        if pd.isna(bucket):
            return None
        bucket = str(bucket).replace("–", "-").replace("_", "-").strip()
        mapping = {
            "5-30": "5_30",
            "30-60": "30_60",
            "60-360": "60_360",
            "30-360": "30_360",
            "90-180": "60_360"  # Treat this as part of same group
        }
        return mapping.get(bucket)

    df['bucket_norm'] = df[bucket_column].apply(normalize_bucket)

    # Prepare output
    results = {}

    # Iterate over segments
    for segment, loan_names in categories.items():
        df_segment = df[df['loan_name'].isin(loan_names)].copy()

        if df_segment.empty:
            logger.info(f"Segment {loan_names} is empty.")
            continue
        logger.info(f"Segmentation {loan_names, segment} processing...")
        # Now split by bucket
        for bucket in df_segment['bucket_norm'].dropna().unique():
            df_sub = df_segment[df_segment['bucket_norm'] == bucket].copy()
            if not df_sub.empty:
                key = f"df_{segment}_{bucket}"
                results[key] = df_sub

    return results


def merge_two_week_buckets(segments_dict):
    """
    Merges TWO_WEEK buckets 30_60 and 60_360 into a unified 30_360 segment.

    Args:
        segments_dict (dict): Dictionary of segment DataFrames (from split_by_segment_and_bucket)

    Returns:
        dict: Updated dictionary with df_TWO_WEEK_30_360 merged and originals removed.
    """

    df_30_60 = segments_dict.get("df_TWO_WEEK_30_60")
    df_60_360 = segments_dict.get("df_TWO_WEEK_60_360")

    if df_30_60 is None and df_60_360 is None:
        logger.info("Nothing to merge: both TWO_WEEK segments are missing.")
        return segments_dict

    # Combine non-null dataframes
    combined = pd.concat(
        [df for df in [df_30_60, df_60_360] if df is not None],
        ignore_index=True
    )

    # Remove the originals
    segments_dict.pop("df_TWO_WEEK_30_60", None)
    segments_dict.pop("df_TWO_WEEK_60_360", None)

    # Insert merged version
    segments_dict["df_TWO_WEEK_30_360"] = combined

    return segments_dict