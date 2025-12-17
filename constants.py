import numpy as np
import string


DT_LABEL = '01042025'

TS_COL = "snap_date"

TARGET = "target" 
TARGET_QUANT = "amount_of_payment"
TARGET_LOG_QUANT = TARGET_QUANT + "_log"

MISSING = "__MISSING__"
OTHER = "__OTHER__"
NAN = np.nan

EPS = 1e-6

CAT = "cat__"
NUM = "num__"
BIN = "__bin"
WOE = "__woe"
SEP = "____"

GROUPS = string.ascii_uppercase     

DT_START = '2024-01-01'
DT_END = '2025-06-01'