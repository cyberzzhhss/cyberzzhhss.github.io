---
title: "Quantconnect Trending System"
excerpt_separator: <!--more-->
tags:
  - Quant
categories:
  - Finance
classes: wide

---

Quantconnect is World's Leading Algorithmic Trading Platform
It provides a free algorithm backtesting tool and financial data so engineers can design algorithmic trading strategies.

The following is a simple <strong>Trending</strong> System.
It is for demo purpose only.

The model uses <strong>[Bollinger Bands](https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands)</strong>(BB) and <strong>[Relative Strength Index](https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi)</strong>(RSI)

Version 1 uses 4-hour aggregates.
It is a Mean reverting strategy where it assumes SPY will revert to its historical mean.

BB uses past 30 data points and 2 standard deviation for bands
RSI uses past 14 data points

The model sells when RSI > 70 AND Price > BB Upper Band

The model buys when RSI < 30 AND Price < BB Lower Band

{% highlight python linenos %}

#original 
class Algo101_Version_1(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)  # Set Start Date
        self.SetEndDate(2020,10,1) #Comment
        self.SetCash(100000)  # Set Strategy Cash
        symbol = "SPY"
        self.ticker = self.AddEquity(symbol,Resolution.Hour).Symbol
        self.rsi = self.RSI(symbol, 14, MovingAverageType.Simple,Resolution.Hour)
        self.bb = self.BB(symbol,30, 2, MovingAverageType.Simple,Resolution.Hour)
        self.RegisterIndicator(symbol, self.rsi, timedelta(hours = 4))
        self.RegisterIndicator(symbol, self.bb, timedelta(hours = 4)) 
        
        self.SetWarmup(timedelta(30))

    def OnData(self, data):
        symbol = "SPY"
        if data[self.ticker] is None: return
        if (not self.rsi.IsReady) or (not self.bb.IsReady): return

        value = data[self.ticker].Value
        
        buy_signal = self.rsi.Current.Value < 30 and value < self.bb.LowerBand.Current.Value
        sell_signal = self.rsi.Current.Value > 70 and value > self.bb.UpperBand.Current.Value
        
        if buy_signal: self.SetHoldings(symbol,1)
        if sell_signal: self.SetHoldings(symbol,-1)
        
        if self.Portfolio.Invested:
            if self.Portfolio[symbol].IsLong and not buy_signal:
                self.Liquidate(symbol)
                
            if self.Portfolio[symbol].IsShort and not sell_signal:
                self.Liquidate(symbol)

{% endhighlight %}
![model1_backtest]({{site.url}}{{site.baseurl}}/assets/images/model1_backtest.png)
![model1_stats]({{site.url}}{{site.baseurl}}/assets/images/model1_stats.png)

The model 1 performs badly.

The idea is to change Mean Reversion Strategy to Momentum Strategy by switching sell and buy signal and modify the RSI period and BollingerBands' period and standard deviation.

Version 2 uses 4-hour aggregates.
It is a momentum strategy where it assumes the price will maintain its momentum for a while

BB uses past 20 data points and 1.5 standard deviation for bands
RSI uses past 10 data points

The model buys when RSI < 40 AND Price < BB Upper Band

The model sells when RSI > 60 AND Price > BB Lower Band


{% highlight python linenos %}
#original 
class Algo101_Version_2(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)  # Set Start Date
        self.SetEndDate(2020,10,1) #Comment
        self.SetCash(100000)  # Set Strategy Cash
        symbol = "SPY"
        self.ticker = self.AddEquity(symbol,Resolution.Hour).Symbol
        self.rsi = self.RSI(symbol, 10, MovingAverageType.Simple,Resolution.Hour)
        self.bb = self.BB(symbol,20, 1.5, MovingAverageType.Simple,Resolution.Hour)
        self.RegisterIndicator(symbol, self.rsi, timedelta(hours = 4))
        self.RegisterIndicator(symbol, self.bb, timedelta(hours = 4)) 
        
        self.SetWarmup(timedelta(30))

    def OnData(self, data):
        symbol = "SPY"
        if data[self.ticker] is None: return
        if (not self.rsi.IsReady) or (not self.bb.IsReady): return

        value = data[self.ticker].Value
        
        sell_signal = self.rsi.Current.Value < 40 and value < self.bb.LowerBand.Current.Value
        buy_signal = self.rsi.Current.Value > 60 and value > self.bb.UpperBand.Current.Value
        
        if buy_signal: self.SetHoldings(symbol,1)
        if sell_signal: self.SetHoldings(symbol,-1)
        
        if self.Portfolio.Invested:
            if self.Portfolio[symbol].IsLong and not buy_signal:
                self.Liquidate(symbol)
                
            if self.Portfolio[symbol].IsShort and not sell_signal:
                self.Liquidate(symbol)

{% endhighlight %}

![model2_backtest]({{site.url}}{{site.baseurl}}/assets/images/model2_backtest.png)
![model2_stats]({{site.url}}{{site.baseurl}}/assets/images/model2_stats.png)


Model 2 has a better return after modification.
