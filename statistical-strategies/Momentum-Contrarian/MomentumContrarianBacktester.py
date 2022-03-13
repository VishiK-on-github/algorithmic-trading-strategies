import pandas as pd
import numpy as np
import plotly.graph_objs as go

class MomentumContrarianBacktester():   
    
    def __init__(self, symbol, start, end, tc, flag):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.flag = flag
        self.results = None
        self.get_data()
        
    def __repr__(self):
        return f"MomentumContrarianBacktester(symbol = {self.symbol}, start = {self.start}, end = {self.end}, flag = {self.flag})"
        
    def get_data(self):
        raw = pd.read_csv("../../resources/intraday.csv", parse_dates = ["time"], index_col = "time")
        raw = raw.Close.to_frame().dropna()
        raw["returns"] = np.log(raw / raw.shift(1))
        raw.dropna(inplace=True)
        self.data = raw
        
    def test_strategy(self, window = 1):
        self.window = window
        data = self.data.copy().dropna()

        if self.flag == 'c':
            data["position"] = -np.sign(data["returns"].rolling(self.window).mean())
        else:
            data["position"] = np.sign(data["returns"].rolling(self.window).mean())

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

            if self.flag == 'c':
                title = f"{self.symbol} | Window = {self.window} | TC = {self.tc} - Contrarian"
            else:
                title = f"{self.symbol} | Window = {self.window} | TC = {self.tc} - Momentum"
                
            fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')

            fig.show()
            
    def optimize_parameter(self, window_range):
        
        windows = range(*window_range)
            
        results = []
        for window in windows:
            results.append(self.test_strategy(window)[0])
        
        best_perf = np.max(results) # best performance
        opt = windows[np.argmax(results)] # optimal parameter
        
        # run/set the optimal strategy
        self.test_strategy(opt)
        
        # create a df with many results
        many_results =  pd.DataFrame(data = {"window": windows, "performance": results})
        self.results_overview = many_results
        
        return opt, best_perf