import pandas as pd
import json
import pandas as pd
from oandapyV20 import API
from oandapyV20.exceptions import V20Error,StreamTerminated
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.contrib.factories import InstrumentsCandlesFactory

import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class DNNTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, lags, model, mu, std, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []
        
        self.accountID = "101-001-21299861-001"
        self.access_token="f4f5eec60b152f8b94314acd730ef791-f94a0597d0f2f930153e604c4c4abfba"
        self.api = API(access_token=self.access_token, environment="practice")

        #*****************add strategy-specific attributes here******************
        self.window = window
        self.lags = lags
        self.model = model
        self.mu = mu
        self.std = std
        #************************************************************************
    
    def get_most_recent(self, days = 5):        
            now = datetime.utcnow()
            now = now - timedelta(microseconds = now.microsecond)
            now = now - timedelta(minutes=1,seconds=30)
            past = now - timedelta(days = days)
            params = {'from':past.strftime('%Y-%m-%dT%H:%M:%SZ'), 'to':now.strftime('%Y-%m-%dT%H:%M:%SZ'),'granularity':'M15'} 
            resList = []
            for R in InstrumentsCandlesFactory(instrument=self.instrument,params=params):
              self.api.request(R)
              resList = resList + R.response.get('candles')
            df = pd.DataFrame(resList) 
            print(df)
            df = pd.DataFrame(list(df['mid']),pd.to_datetime(df['time'])) 
            
            df.rename(columns = {"c":self.instrument}, inplace = True)
            print(df)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            print ("TPQOA Last:",self.last_bar)   
            s = PricingStream(accountID=self.accountID, params={"instruments":self.instrument})
            try:
               n = 0
               for R in self.api.request(s):
                  print (R['time'], R['type'])
                  if R['type'] == 'PRICE':
                    recent_tick = pd.to_datetime(R['time']) # Pandas Timestamp Object

                    df = pd.DataFrame({self.instrument:(pd.to_numeric(R['closeoutBid']) + pd.to_numeric(R['closeoutAsk']))/2}, 
                          index = [recent_tick])
                    df = self.raw_data.append(df)
                    df = df.resample(self.bar_length, label = "right").last().iloc[:-1]
                    self.raw_data = df.copy()
                    self.last_bar = self.raw_data.index[-1]
             
                    n += 1
                    if n > 1:
                       s.terminate("Completed!")
            except StreamTerminated as e:
               print("last bar done!")
            except V20Error as e:
               print("Error: {}".format(e))
            print ("OPY Last:",self.last_bar)    

    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                                  label="right").last().ffill().iloc[:-1])
        print ("Resample 1:", self.raw_data)
        self.tick_data = self.tick_data.iloc[-1:]
        print ("Resample 2:", self.tick_data)
        self.last_bar = self.raw_data.index[-1]
        print ("Resample 3:", self.last_bar)

    def process_stream(self, instruments, stopAt):
        s = PricingStream(accountID=self.accountID, params={"instruments":instruments})
        try:
          n = 0
          for R in self.api.request(s):
             print (R['time'], R['type'])
             if R['type'] == 'PRICE':
                   recent_tick = pd.to_datetime(R['time']) # Pandas Timestamp Object

                   df = pd.DataFrame({'instrument':(pd.to_numeric(R['closeoutBid']) + pd.to_numeric(R['closeoutAsk']))/2}, 
                          index = [recent_tick])
                   self.tick_data = self.tick_data.append(df)

                   if recent_tick - self.last_bar > self.bar_length:
                       self.resample_and_join()
                       self.define_strategy()
                       self.execute_trades()
             
                   n += 1
             if n > stopAt:
                s.terminate("Completed!")

        except StreamTerminated as e:
            print("Stopping!")
        except V20Error as e:
            print("Error: {}".format(e))
                
    def on_success(self, time, bid, ask):
        print(self.ticks, end = " ")
        
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick])
        self.tick_data = self.tick_data.append(df)
        
        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
    
 
    
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        #create features
        df = df.append(self.tick_data) # append latest tick (== open price of current bar)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["dir"] = np.where(df["returns"] > 0, 1, 0)
        df["sma"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
        df["boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[self.instrument].rolling(self.window).std()
        df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(self.window).std()
        df.dropna(inplace = True)
        
        # create lags
        self.cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                self.cols.append(col)
        df.dropna(inplace = True)
        
        # standardization
        df_s = (df - self.mu) / self.std
        # predict
        print ("Predict params:", df_s[self.cols])
        df["proba"] = self.model.predict(df_s[self.cols])
        
        #determine positions
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["position"] = np.where(df.proba < 0.47, -1, np.nan)
        df["position"] = np.where(df.proba > 0.53, 1, df.position)
        df["position"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")

if __name__ == "__main__":
    
    import keras
    model = keras.models.load_model("DNN_model")
    import pickle
    params = pickle.load(open("params.pkl", "rb"))
    mu = params["mu"]
    std = params["std"]
    trader = DNNTrader(r"C:\Users\rharidas\MachineLearningProject-main\Part4_Materials\Part4_Materials\Oanda\oanda.cfg",  "EUR_USD", bar_length = "20min",
                   window = 50, lags = 5, model = model, mu = mu, std = std, units = 100000)
    trader.get_most_recent()
    trader.process_stream(trader.instrument, stopAt = 100)
    if trader.position != 0: 
        close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units, 
                                          suppress = True, ret = True) 
        trader.report_trade(close_order, "GOING NEUTRAL")
        trader.position = 0