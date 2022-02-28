import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import datetime

import config
from Preprocess import FeatureEngineer, data_split
from Environment_Custom import StockTradingEnv
from Environment_CustomX import StockTradingEnvX
from GetData import getdata

from Agent_Custom import DRLAgent
import pandas as pd
import yfinance as yf

from customer import customer



import config
from Preprocess import FeatureEngineer

from customer import customer

tickers = config.TICKER_LIST
date1=config.START_DATE
date2=config.END_DATE
list_data = []
for ticker in tickers:
    data = yf.download(tickers=ticker , start=date1, end=date2, interval='1d' )
    data["tic"] = ticker
    data["date"] = data.index
    data = data.reset_index()
    list_data.append(data)


df = pd.concat(list_data)
del df['Date']



df.isna().sum()
multi_ticker_df = df.ffill().bfill()
print(multi_ticker_df)

multi_ticker_df.columns= multi_ticker_df.columns.str.strip().str.lower()
multi_ticker_df.columns

multi_ticker_df.sort_values(['date','tic'],ignore_index=True).head()



print("==============Start Feature Engineering===========")
tech_indicator_list=config.TECHNICAL_INDICATORS_LIST

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)

#vix volatilite endeksi

processed = fe.preprocess_data(multi_ticker_df)



# processed_full = processed.sort_values(['date','tic'])
# processed_full= processed_full.ffill().bfill()
# print(processed_full)
