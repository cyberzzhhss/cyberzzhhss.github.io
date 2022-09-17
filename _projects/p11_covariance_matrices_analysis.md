---
title: "Covariance Matrix Estimators Analysis"
excerpt: "Compare the performance of different covariance matrix estimators for out-of-sample volatility of the Markowitz portfolio"
last_modified_at: 2022-4-22
header:
  teaser: /assets/images/cov_matrices.png
---

Comparison of the optimal shrinkage function that maps the empirical eigenvalue $\lambda_i$ onto 'cleaned' version, $\xi_i$


![image1]({{site.url}}{{site.baseurl}}/assets/images/cov_matrices.png)

Formula 


![image2]({{site.url}}{{site.baseurl}}/assets/images/rie_estimator.png)

The question of building reliable estimators of covariance or of correlation matrices has a long history in finance, and more generally in multivariate statistical analysis. The performance of any mean-variance optimization scheme closely depends on the variance-covariance matrices. Therefore, it is of utmost importance to select a suitable metric against which candidate methods of covariance matrix estimators can be tested. In this section, I compare three commonly used covariance matrix estimators – the empirical covariance, the “eigenvalue clipping” covariance estimator, and the “optimal shrinkage” estimator by the out-of-sample volatility of the Markowitz portfolios built from each estimator.

[My github project](https://github.com/cyberzzhhss/covariance_matrix_estimation)


[My detailed report](https://github.com/cyberzzhhss/covariance_matrix_estimation/blob/master/report.pdf)