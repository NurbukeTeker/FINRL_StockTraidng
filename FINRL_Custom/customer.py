import config
import yfinance as yf

def calculate_initial_cash(initial_amount,initialStocks ):
    data = yf.download(config.TICKER_LIST ,start= config.START_TRADE_DATE, end= "2020-07-28")['Adj Close']
    adjclose_values = list(data.iloc[0])
    dotproduct=0
    for a,b in zip(adjclose_values,initialStocks):
        dotproduct = dotproduct+a*b
    print('Dot product is:', dotproduct)

    initial_amount = initial_amount - dotproduct
    return initial_amount

class Customer():
    def __init__(self, totalAmount,initialStocks):
        self.totalAmount= totalAmount
        self.initialStocks = initialStocks
        self.hmax = self.totalAmount /1000

#customers intiail stock values


def getInitialStock(stock_type):
    if stock_type =="DOW":
        # tic_index = dict(zip(config.DOW_30_TICKER,len(config.DOW_30_TICKER)*[0])) 
        # initialStocks = list(tic_index.values())[:8]
        tic_index = {
            'AAPL':0, 'CAT':0, 'HD':0, 'IBM':0, 'JNJ':0, 'MRK':0, 'NKE':0, 'UNH':0
        }
        initialStocks = list(tic_index.values())
        return initialStocks
            
    else:
        tic_index = {"VESTL.IS":0,
            "VAKBN.IS":0,
            "TUPRS.IS":0,
            "THYAO.IS":0,
            "HALKB.IS":0,
            "SASA.IS":0,
            "PETKM.IS":0,
            "SAHOL.IS":0,
            "PGSUS.IS":0,
            "GARAN.IS":0,
            "AKBNK.IS":0,
            "ARCLK.IS":0,
            "BIMAS.IS":0,
            "EREGL.IS":0}
        initialStocks = list(tic_index.values())
        return initialStocks


customer = Customer(1000000 , getInitialStock("DOW"))
customer.totalAmount = calculate_initial_cash(customer.totalAmount,customer.initialStocks)
print("Customer Final Look---------")
print(customer.totalAmount)
print(customer.hmax)
#customer created

