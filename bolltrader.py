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

class BollTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, SMA,SMA_L0,SMA_U0,SMA_L1,SMA_U1,SMA_L2,SMA_U2, dev, units):
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
        self.SMA_L0 = SMA_L0
        self.SMA_U0 = SMA_U0
        self.SMA_L1 = SMA_L1
        self.SMA_U1 = SMA_U1
        self.SMA_L2 = SMA_L2
        self.SMA_U2 = SMA_U2
        self.SMA = SMA
        self.dev = dev
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

                    df = pd.DataFrame({'instrument':(pd.to_numeric(R['closeoutBid']) + pd.to_numeric(R['closeoutAsk']))/2}, 
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
    
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                                  label="right").last().ffill().iloc[:-1])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        df["SMA_L0"] = df[self.instrument].rolling(self.SMA_L0).mean()
        df["SMA_U0"] = df[self.instrument].rolling(self.SMA_U0).mean()
        df["SMA_L1"] = df[self.instrument].rolling(self.SMA_L1).mean()
        df["SMA_U1"] = df[self.instrument].rolling(self.SMA_U1).mean()
        df["SMA_L2"] = df[self.instrument].rolling(self.SMA_L2).mean()
        df["SMA_U2"] = df[self.instrument].rolling(self.SMA_U2).mean()
        df["SMA"] = df[self.instrument].rolling(self.SMA).mean()
        df["Lower"] = df["SMA"] - df[self.instrument].rolling(self.SMA).std() * self.dev
        df["Upper"] = df["SMA"] + df[self.instrument].rolling(self.SMA).std() * self.dev
        df["distance"] = df[self.instrument] - df.SMA
        df["position"] = np.where(df[self.instrument] < df.Lower, 1, np.nan)
        df["position"] = np.where(df[self.instrument] > df.Upper, -1, df["position"])
        df["position"] = np.where(df.distance * df.distance.shift(1) < 0, 0, df["position"])    
        df["position"] = df.position.ffill().fillna(0)   
        
        if (df[self.instrument].all() > df["Lower"].all()) and (df[self.instrument].all() < df["Upper"].all()) and (df["distance"].abs().all() < 0.5):
          sma_strat0 = np.where(df["SMA_L0"] > df["SMA_U0"], 1, -1)
          sma_strat1 = np.where(df["SMA_L1"] > df["SMA_U1"], 1, -1)
          sma_strat2 = np.where(df["SMA_L2"] > df["SMA_U2"], 1, -1)
          sma_strat1 = np.sign(sma_strat0 + sma_strat1)
          sma_strat = np.sign(sma_strat1 + sma_strat2)        
          df["position"] = sma_strat
          df["position"] = df.position.ffill().fillna(0)
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
        
    trader = BollTrader(r"C:\Users\rharidas\MachineLearningProject-main\Part4_Materials\Part4_Materials\Oanda\oanda.cfg", "EUR_USD", "1min",SMA=20,SMA_L0=3,SMA_U0=5,SMA_L1=5,SMA_U1=9,SMA_L2=9,SMA_U2=20, dev = 2, units = 100000)
    trader.get_most_recent()
    trader.process_stream(trader.instrument, stopAt = 200)
    if trader.position != 0: 
        close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units, 
                                          suppress = True, ret = True) 
        trader.report_trade(close_order, "GOING NEUTRAL")
        trader.position = 0