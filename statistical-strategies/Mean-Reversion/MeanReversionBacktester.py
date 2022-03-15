import pandas as pd
import numpy as np
from itertools import product
import plotly.graph_objs as go

class MeanReversionBacktester():

  def __init__(self, symbol, SMA, dev, start, end, tc):

    self.symbol = symbol
    self.SMA = SMA
    self.dev = dev
    self.start = start
    self.end = end
    self.tc = tc
    self.results = None
    self.get_data()
    self.prepare_data()

  def __repr__(self):

    return f'MeanReversionBacktester(symbol = {self.symbol}, SMA = {self.SMA}, dev = {self.dev}, start = {self.start}, end = {self.end}, tc={self.tc})'

  def get_data(self):

    if self.symbol == 'EUR/USD':

      raw = pd.read_csv(filepath_or_buffer='../../resources/intraday.csv', parse_dates=['time'], index_col='time')
      raw = raw.Close.to_frame().dropna()
      raw = raw[self.start:self.end]
      raw["returns"] = np.log(raw / raw.shift(1))
      self.data = raw

    # Maybe define an else case to get data from yahoo finance

  def prepare_data(self):

    data = self.data.copy()
    data['SMA'] = data['Close'].rolling(self.SMA).mean()
    data['Lower'] = data['SMA'] - data['Close'].rolling(self.SMA).std() * self.dev
    data['Upper'] = data['SMA'] + data['Close'].rolling(self.SMA).std() * self.dev
    data.dropna(inplace=True)
    self.data = data

  def set_parameters(self, SMA = None, dev = None):
        
    if SMA is not None:
        self.SMA = SMA
        self.data["SMA"] = self.data["Close"].rolling(self.SMA).mean()
        self.data["Lower"] = self.data["SMA"] - self.data["Close"].rolling(self.SMA).std() * self.dev
        self.data["Upper"] = self.data["SMA"] + self.data["Close"].rolling(self.SMA).std() * self.dev
        
    if dev is not None:
        self.dev = dev
        self.data["Lower"] = self.data["SMA"] - self.data["Close"].rolling(self.SMA).std() * self.dev
        self.data["Upper"] = self.data["SMA"] + self.data["Close"].rolling(self.SMA).std() * self.dev

  def test_strategy(self):

    data = self.data.copy().dropna()
    data["distance"] = data.Close - data.SMA
    data["position"] = np.where(data.Close < data.Lower, 1, np.nan)
    data["position"] = np.where(data.Close > data.Upper, -1, data["position"])
    data["position"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position"])
    data["position"] = data.position.ffill().fillna(0)
    data["strategy"] = data.position.shift(1) * data["returns"]
    data.dropna(inplace=True)
    
    # determine the number of trades in each bar
    data["trades"] = data.position.diff().fillna(0).abs()
    
    # subtract transaction/trading costs from pre-cost return
    data['strategy_net'] = data.strategy - data.trades * self.tc
    
    data["creturns"] = data["returns"].cumsum().apply(np.exp)
    data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
    data["cstrategy_net"] = data["strategy_net"].cumsum().apply(np.exp)
    self.results = data
    
    perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
    outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
    
    return round(perf, 6), round(outperf, 6)

  def plot_results(self):

    if self.results is None:
      print("Run test_strategy() first")

    else:
      fig = go.Figure()

      fig.add_trace(go.Scatter(x=self.results.index, y=self.results.creturns, name='Returns (Base)'))
      fig.add_trace(go.Scatter(x=self.results.index, y=self.results.cstrategy, name='Returns (Strategy)'))
      fig.add_trace(go.Scatter(x=self.results.index, y=self.results.cstrategy_net, name='Returns (Strategy + cost)'))

      title = f"{self.symbol} | SMA = {self.SMA} | dev = {self.dev} | TC = {self.tc}"
      fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')

      fig.show()

  def optimize_parameters(self, SMA_range, dev_range):

    combinations = list(product(range(*SMA_range), range(*dev_range)))
    
    # test all combinations
    results = []
    for comb in combinations:
        self.set_parameters(comb[0], comb[1])
        results.append(self.test_strategy()[0])
    
    best_perf = np.max(results) # best performance
    opt = combinations[np.argmax(results)] # optimal parameters
    
    # run/set the optimal strategy
    self.set_parameters(opt[0], opt[1])
    self.test_strategy()
                
    # create a df with many results
    many_results =  pd.DataFrame(data = combinations, columns = ["SMA", "dev"])
    many_results["performance"] = results
    self.results_overview = many_results
                        
    return opt, best_perf