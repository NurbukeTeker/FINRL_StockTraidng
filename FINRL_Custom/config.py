import pandas as pd
from feature_extractors import RNNFeatureExtractor

TOTAL_TSTP =  10000
TICKER_LIST = ["VESTL.IS","VAKBN.IS","TUPRS.IS","THYAO.IS","HALKB.IS","SASA.IS","PETKM.IS","SAHOL.IS","PGSUS.IS","GARAN.IS","AKBNK.IS","ARCLK.IS","BIMAS.IS","EREGL.IS"] # 14 tane hisse
## time_fmt = '%Y-%m-%d'
START_DATE = "2015-07-27"
END_DATE = "2021-07-28"

START_TRADE_DATE =  "2020-07-27"
END_TRADE_DATE_3ay = "2020-10-28"
END_TRADE_DATE_6ay = "2021-01-28"
END_TRADE_DATE_9ay = "2021-04-28"

START_DATE_COR = "2015-07-27"
START_DATE_TRADE_COR = "2019-01-01"
END_DATE_COR = "2020-01-01"


TRAINED_MODEL_DIR = f"trained_models"

DATA_SAVE_DIR = f"datasets"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"

## dataset default columns
DEFAULT_DATA_COLUMNS = ["date", "tic", "close"]

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
#TECHNICAL_INDICATORS_LIST = ["macd","boll_ub","boll_lb","rsi_30", "cci_30", "dx_30","close_30_sma","close_60_sma"]
TECHNICAL_INDICATORS_LIST = ["macd","rsi_30", "cci_30", "dx_30"]
TECHNICAL_INDICATORS_LIST2 = ["bar_hc","bar_ho","bar_hl","bar_cl","bar_ol","bar_co"]#,"adj_open","adj_high","adj_low"]#,"bar_mov" ,]

## Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 64,
    "ent_coef": "auto_0.1",
}

POLICY_KWARGS = {
    "net_arch": [400, 300],
    "features_extractor_class": RNNFeatureExtractor
    }


DOW_30_TICKER = [
    "AAPL",
    "MSFT",
    "JPM",
    "RTX",
    "PG",
    "GS",
    "NKE",
    "DIS",
    "AXP",
    "HD",
    "INTC",
    "WMT",
    "IBM",
    "MRK",
    "UNH",
    "KO",
    "CAT",
    "TRV",
    "JNJ",
    "CVX",
    "MCD",
    "VZ",
    "CSCO",
    "XOM",
    "BA",
    "MMM",
    "PFE",
    "WBA",
    "DD",
]