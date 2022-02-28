import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import datetime

import config
from Preprocess import FeatureEngineer, data_split
# from Environment_Custom import StockTradingEnv
from EnvironmentOhlcv import StockTradingEnv
from GetData import getdata,read_proceesed,getdow30data_minute

from Agent_Custom import DRLAgent
import pandas as pd
import yfinance as yf

from customer import customer


#part of multi-stock data get
processed_full = getdow30data_minute()
processed_full = getdata() 
# processed_full = read_proceesed() #read quickly from processed_full.csv


#Train Test
train = data_split(processed_full,config.START_DATE, config.END_TRADE_DATE_6ay)
trade = data_split(processed_full, config.END_TRADE_DATE_6ay, config.END_DATE)

#saatlik data için:
train = data_split(processed_full,"2020-01-01", "2020-10-01")
trade = data_split(processed_full, "2020-10-01", "2021-01-01")



config.TECHNICAL_INDICATORS_LIST = config.TECHNICAL_INDICATORS_LIST + config.TECHNICAL_INDICATORS_LIST2





#Create Env
stock_dimension = len(train.tic.unique()) # number of tickers
state_space = (1 + 2 * stock_dimension + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension)

env_kwargs = {
        "hmax": customer.hmax, 
        "initial_amount": customer.totalAmount , 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4}


e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()


#Create Model
agent = DRLAgent(env=env_train)


#Train Part
print("==============Model Training===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

model_a2c = agent.get_model("a2c")
trained_a2c = agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=config.TOTAL_TSTP )

# trained_sac.save('A2C_model.h5') 

#Trade Part
print("==============Start Trading===========")
e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250, **env_kwargs)

df_account_value, df_actions = DRLAgent.DRL_prediction( model=trained_a2c, environment = e_trade_gym)

# print(df_actions)
# print(df_account_value)


with open("df_account_value_BIST_a2c_.csv", 'a') as f1:
    df_account_value.to_csv(f1, header=False)

# with open("df_actions_BIST_a2c.csv", 'a') as f2:
#     df_actions.to_csv(f2, header=False)




from BackTesting import backtest_stats ,VisualizeResult

VisualizeResult(df_account_value, "A2C_")

#BackTesting Part
print("==============Get Backtest Results===========")
perf_stats_all = backtest_stats(df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("perf_stats_all_A2C_.csv")

print(perf_stats_all)
