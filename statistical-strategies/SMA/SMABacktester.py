import pandas as pd
import numpy as np
from itertools import product
import plotly.graph_objs as go

class SMABacktester():
    
    def __init__(self, symbol, SMA_S, SMA_L, start, end,tc):
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.tc=tc
        self.results = None 
        self.get_data()
        self.prepare_data()
        
    def __repr__(self):
  
        return f'SMABacktester(symbol = {self.symbol}, SMA_S = {self.SMA_S},  SMA_L = {self.SMA_L}, start = {self.start}, end = {self.end}, tc={self.tc})'
        
    def get_data(self):
        raw = pd.read_csv("../../resources/intraday.csv", parse_dates = ["time"], index_col = "time")
        raw = raw.Close.to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
        
    def prepare_data(self):
        data = self.data.copy()
        data["SMA_S"] = data["Close"].rolling(self.SMA_S).mean()
        data["SMA_L"] = data["Close"].rolling(self.SMA_L).mean()
        self.data = data
        
    def set_parameters(self, SMA_S = None, SMA_L = None):
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["Close"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["Close"].rolling(self.SMA_L).mean()
            
    def test_strategy(self):
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
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
            print("Run test_strategy() first.")
        else:
          fig = go.Figure()

          fig.add_trace(go.Scatter(x=self.results.index, y=self.results.creturns, name='Returns (Base)'))
          fig.add_trace(go.Scatter(x=self.results.index, y=self.results.cstrategy, name='Returns (Strategy)'))
          fig.add_trace(go.Scatter(x=self.results.index, y=self.results.cstrategy_net, name='Returns (Strategy + cost)'))

          title = f"{self.symbol} | SMA_L = {self.SMA_L} | SMA_S = {self.SMA_S} | TC = {self.tc}"
          fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')

          fig.show()
    
    def optimize_parameters(self, SMA_S_range, SMA_L_range):
        combinations = list(product(range(*SMA_S_range), range(*SMA_L_range)))
        
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
        many_results =  pd.DataFrame(data = combinations, columns = ["SMA_S", "SMA_L"])
        many_results["performance"] = results
        self.results_overview = many_results
                            
        return opt, best_perf