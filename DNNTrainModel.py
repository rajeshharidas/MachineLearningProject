import pandas as pd
import numpy as np

from DNNModel import *
import matplotlib.pyplot as plt
plt.style.use("seaborn")
pd.set_option('display.float_format', lambda x: '%.5f' % x)

data = pd.read_csv("DNN_data.csv", parse_dates = ["time"], index_col = "time")
symbol = data.columns[0]
data["returns"] = np.log(data[symbol] / data[symbol].shift())
window = 50

df = data.copy()
df["dir"] = np.where(df["returns"] > 0, 1, 0)
df["sma"] = df[symbol].rolling(window).mean() - df[symbol].rolling(150).mean()
df["boll"] = (df[symbol] - df[symbol].rolling(window).mean()) / df[symbol].rolling(window).std()
df["min"] = df[symbol].rolling(window).min() / df[symbol] - 1
df["max"] = df[symbol].rolling(window).max() / df[symbol] - 1
df["mom"] = df["returns"].rolling(3).mean()
df["vol"] = df["returns"].rolling(window).std()
df.dropna(inplace = True)

lags = 5

cols = []
features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

for f in features:
        for lag in range(1, lags + 1):
            col = "{}_lag_{}".format(f, lag)
            df[col] = df[f].shift(lag)
            cols.append(col)
df.dropna(inplace = True)

split = int(len(df)*0.66)
split

train = df.iloc[:split].copy()
train

test = df.iloc[split:].copy()
test

mu, std = train.mean(), train.std() # train set parameters (mu, std) for standardization

train_s = (train - mu) / std # standardization of train set features


# fitting a DNN model with 3 Hidden Layers (50 nodes each) and dropout regularization

set_seeds(100)
model = create_model(hl = 3, hu = 50, dropout = True, input_dim = len(cols))
model.fit(x = train_s[cols], y = train["dir"], epochs = 50, verbose = False,
          validation_split = 0.2, shuffle = False, class_weight = cw(train))

model.evaluate(train_s[cols], train["dir"]) # evaluate the fit on the train set

pred = model.predict(train_s[cols]) # prediction (probabilities)
pred

test_s = (test - mu) / std # standardization of test set features (with train set parameters!!!)

model.evaluate(test_s[cols], test["dir"])

pred = model.predict(test_s[cols])
pred

test["proba"] = model.predict(test_s[cols])
test["position"] = np.where(test.proba < 0.47, -1, np.nan) # 1. short where proba < 0.47
test["position"] = np.where(test.proba > 0.53, 1, test.position) # 2. long where proba > 0.53
test.index = test.index.tz_localize("UTC")
test["NYTime"] = test.index.tz_convert("America/New_York")
test["hour"] = test.NYTime.dt.hour
test["position"] = np.where(~test.hour.between(2, 12), 0, test.position) # 3. neutral in non-busy hours
test["position"] = test.position.ffill().fillna(0) # 4. in all other cases: hold position
test.position.value_counts(dropna = False)
test["strategy"] = test["position"] * test["returns"]
test["creturns"] = test["returns"].cumsum().apply(np.exp)
test["cstrategy"] = test["strategy"].cumsum().apply(np.exp)
ptc = 0.000059
test["trades"] = test.position.diff().abs()
test["strategy_net"] = test.strategy - test.trades * ptc
test["cstrategy_net"] = test["strategy_net"].cumsum().apply(np.exp)
model.save("DNN_model")

import pickle
params = {"mu":mu, "std":std}

pickle.dump(params, open("params.pkl", "wb"))