# doji-analysis

The candlestick chart is an important technical analysis tool, used by traders of different financial products. Among a variety of techniques, the investigation
of specific candle formats is widely used trying to evaluate the market bias, therefore predicting future price movements. 

One of this candle formats is known as 'doji', and it consists in fairly simmetrical wicks and a small body size. 

In this project two different machine learning models were used to predict if the candle following a doji (under specific conditions) would close after a determined target (take profit), and if that would happen without hitting a determined stop loss. 

To do that, in the first case (Doji_ML.py) data from 200.000 candles from 2020 was used, both for training and testing (splitted data). Secondly (Doji_ML_v1.py), the previous data was used only to train the models, and a separate source (GBPUSD_M15_10k_2021.csv) used to perform the testings. 

All results are shown within the code.


