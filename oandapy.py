import json
import pandas as pd
from oandapyV20 import API
from oandapyV20.exceptions import V20Error,StreamTerminated
from oandapyV20.endpoints.pricing import PricingStream

accountID = "101-001-21299861-001"
access_token="f4f5eec60b152f8b94314acd730ef791-f94a0597d0f2f930153e604c4c4abfba"

api = API(access_token=access_token, environment="practice")
df = pd.DataFrame()

instruments = "EUR_USD"
s = PricingStream(accountID=accountID, params={"instruments":instruments})
try:
    n = 0
    for R in api.request(s):
        print (R['time'], R['type'])
        if R['type'] == 'PRICE':
             recent_tick = pd.to_datetime(R['time']) # Pandas Timestamp Object

             df = df.append(pd.DataFrame({'price':(pd.to_numeric(R['closeoutBid']) + pd.to_numeric(R['closeoutAsk']))/2}, 
                          index = [recent_tick]))
             
             n += 1
        if n > 10:
            s.terminate("Completed!")

except StreamTerminated as e:
    print("Stopping!")
except V20Error as e:
    print("Error: {}".format(e))
print('Data frame: \n')
print(df)