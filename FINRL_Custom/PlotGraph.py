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
from Environment_CustomX import StockTradingEnvX
from GetData import getdata

from Agent_Custom import DRLAgent
import pandas as pd
import yfinance as yf

from customer import customer

#part of multi-stock data get
processed_full = getdata() 

trade = data_split(processed_full, config.START_TRADE_DATE, config.END_DATE)



trade.columns
big_list = []
for i in range(len(config.TICKER_LIST)):
    in_list =[]
    for j,line  in trade.iterrows() : 
        if line["tic"] == config.TICKER_LIST[i]: 
            # print(line["tic"],round(line["adj close"],2))
            in_list.append(round(line["adj close"],2))
    big_list.append(in_list)

big_list[1]

dates= trade["date"].drop_duplicates().reset_index(drop=True)
date_list =  []
for i in dates:
   date_list.append(str(i).split(" ")[0] ) 


len(date_list)


import matplotlib.pyplot as plt
x = date_list
y = big_list

plt.figure(figsize=(15, 15))

plt.title("A test graph")

plt.subplot(14, 1, 1)
plt.plot(x,y[0])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[0])

plt.subplot(14, 1, 2)
plt.plot(x,y[1])
plt.axvline(x=x[65])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[1])


plt.subplot(14, 1, 3)
plt.plot(x,y[2])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[2])


plt.subplot(14, 1, 4)
plt.plot(x,y[3])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[3])



plt.subplot(14, 1, 5)
plt.plot(x,y[4])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[4])



plt.subplot(14, 1, 6)
plt.plot(x,y[5])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[5])



plt.subplot(14, 1, 7)
plt.plot(x,y[6])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[6])


plt.subplot(14, 1, 8)
plt.plot(x,y[7])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[7])



plt.subplot(14, 1, 9)
plt.plot(x,y[8])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[8])



plt.subplot(14, 1, 10)
plt.plot(x,y[9])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[9])



plt.subplot(14, 1, 11)
plt.plot(x,y[10])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[10])


plt.subplot(14, 1, 12)
plt.plot(x,y[11])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[11])


plt.subplot(14, 1, 13)
plt.plot(x,y[12])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[12])



plt.subplot(14, 1, 14)
plt.plot(x,y[13])
plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')
plt.ylabel(config.TICKER_LIST[13])



plt.savefig('result.png')
plt.close()


##############

len(date_list)

import matplotlib.pyplot as plt
interval_date_list = []
for i in range(len(date_list)):
    if i % 7 ==0 :
       interval_date_list.append(date_list[i]) 
    else:
        interval_date_list.append("")

x = date_list
y = big_list
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("A test graph")
plt.figure(figsize=(20,10))
for i in range(len(y)):
    plt.plot(x,y[i],label = 'id %s'%(config.TICKER_LIST[i]))


plt.axvline(x=x[65],color='r')
plt.axvline(x=x[129],color='r')
plt.axvline(x=x[192],color='r')


plt.legend()
plt.show()
plt.savefig('result222.png')
plt.close()
