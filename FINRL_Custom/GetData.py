import yfinance as yf
import config
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")

import config
from Preprocess import FeatureEngineer
from process_data import FeatureExtractor
from customer import customer

def prefill(multi_ticker_df)  :
    df = multi_ticker_df.copy()
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index = 'date',columns = 'tic', values = 'close')

    print(merged_closes.isna().sum())
    merged_closes = merged_closes.ffill().bfill()

    # print(merged_closes.isna().sum())
    # print(merged_closes)
    tics = merged_closes.columns
    df = df[df.tic.isin(tics)]
    df = df.reset_index()
    del df['index']
    return df

def getdata():
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



    multiticker_df = prefill(multi_ticker_df)

    multiticker_df = multiticker_df[multiticker_df.volume != 0]

    print(multiticker_df)

    print("==============Start Feature Engineering===========")
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicator_list,
        use_turbulence=True,
        user_defined_feature=False,
    )


    processed = fe.preprocess_data(multiticker_df)

    processed_full = processed.sort_values(['date','tic'])
    processed_full= processed_full.ffill().bfill()
    print(processed_full)
    #processed_full.to_csv("processed_full.csv")
    extractor = FeatureExtractor(processed_full)

    df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
   # df = extractor.add_adj_features()
    df.dropna(inplace=True) # drops Nan rows
    return df


def read_proceesed():
    df = pd.read_csv("processed_full.csv")
    df.reset_index()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


    extractor = FeatureExtractor(df)

    df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
    df = extractor.add_adj_features()
    df.dropna(inplace=True) # drops Nan rows
    return df



def getdow30data_minute():
    tickers = config.DOW_30_TICKER
    date1="2020-01-01"
    date2="2021-01-01"
    list_data = []
    for ticker in tickers:
        data = yf.download(tickers=ticker , start=date1, end=date2, interval='1h' )
        data["tic"] = ticker
        data["date"] = data.index
        data = data.reset_index()
        list_data.append(data)


    df = pd.concat(list_data)
  
    df.isna().sum()
    multi_ticker_df = df.ffill().bfill()
    print(multi_ticker_df)

    multi_ticker_df.columns= multi_ticker_df.columns.str.strip().str.lower()
    multi_ticker_df.columns



    multiticker_df = prefill(multi_ticker_df)

    multiticker_df = multiticker_df[multiticker_df.volume != 0]

    print(multiticker_df)

    print("==============Start Feature Engineering===========")
    tech_indicator_list=config.TECHNICAL_INDICATORS_LIST

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicator_list,
        use_turbulence=True,
        user_defined_feature=False,
    )


    processed = fe.preprocess_data(multiticker_df)

    processed_full = processed.sort_values(['date','tic'])
    processed_full= processed_full.ffill().bfill()
    print(processed_full)
    processed_full.to_csv("dow_30_processed_full.csv")

    extractor = FeatureExtractor(processed_full)
    df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
    # df = extractor.add_adj_features()
    df.dropna(inplace=True) # drops Nan rows
    return df