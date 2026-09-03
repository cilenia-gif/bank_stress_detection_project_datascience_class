# Bank Stress Detection

This project examines whether publicly available stock market data from major U.S. banks can provide early signals of stress in the banking sector.

I construct a bank equity stress index using rolling volatility and cross-bank correlation measures based on daily stock returns. I then use lagged values of the index in a logistic regression model to examine whether they can help identify future stress periods. The model is evaluated using a time-based train-test split to avoid look-ahead bias and better reflect a real-time setting.

The aim is not to predict financial crises, but to assess whether relatively simple market-based information can provide useful early-warning signals of elevated banking stress.

**Methods:** stress index construction, rolling volatility and correlation measures, logistic regression, classification, time-based out-of-sample validation, robustness checks

**Tools:** Python, Jupyter, pandas, scikit-learn, yfinance

This repository contains the original version of the project submitted as part of a university data science course.

More details are available in the [`PROJECT_README.md`](PROJECT_README.md) and the [final report](project_report.pdf).
