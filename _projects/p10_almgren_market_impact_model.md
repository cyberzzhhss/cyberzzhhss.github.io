---
title: "Almgren Chriss Market Impact Model"
excerpt: "Implementing and Calibrating Almgren-Chriss Market Impact Model"
last_modified_at: 2022-5-20
header:
  teaser: /assets/images/trade_activity.png
---

Liquidation Paths

![image1]({{site.url}}{{site.baseurl}}/assets/images/liquidation_paths.png)

Price Impact 

![image2]({{site.url}}{{site.baseurl}}/assets/images/price_impact.png)

Limit Order Book

![image3]({{site.url}}{{site.baseurl}}/assets/images/trade_activity.png)

Realized Cost

![image4]({{site.url}}{{site.baseurl}}/assets/images/realized_cost
.png)


In thie project, we worked with more than 100GB+ 3-month millisecond-level high-frequency NYSE trades and quotes tick data of more than 1000 tickers to calibrate the Almgren market impact model by applying nonlinear regression.

We also formulated the Almgren-Chriss optimal execution problem as a stochastic control problem and derived the Hamilton–Jacobi–Bellman equation and solved for the control and value function.

[My github project](https://github.com/cyberzzhhss/market_impact_model)


[My detailed report](https://github.com/cyberzzhhss/market_impact_model/blob/master/report.pdf)