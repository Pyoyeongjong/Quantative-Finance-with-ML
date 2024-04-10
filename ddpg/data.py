import pandas as pd
import numpy as np
import time
import os

from sklearn.preprocessing import MinMaxScaler

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

scaler = MinMaxScaler()


scale_cols = ['open', 'high', 'low', 'close',
              'volume', 'sma7', 'sma20', 'sma60',
              'sma120', 'rsi', 'vol_sma', 'upperband',
              'lowerband', 'atr', 'macd', 'macdsignal', 'macdhist',
              'cci', 'adx']

cost_cols = ['open', 'high', 'low', 'close', 'sma7', 'sma20', 'sma60', 'sma120']

volume_cols = ['volume', 'vol_sma']


year = [2021, 2022, 2023]

def load_data_1m(ticker):
    file_list = []
    for y in year:
        file_list.append(f"candle_data/{ticker}_1m_{y}_0_sub.csv")
        file_list.append(f"candle_data/{ticker}_1m_{y}_1_sub.csv")
    df = [pd.read_csv(file).drop(columns='Unnamed: 0').dropna() for file in file_list if os.path.exists(file)]
    df_combine = pd.concat(df, ignore_index=True)

    return df_combine


class Data:
    def __init__(self):
        self.data_1w = None
        self.data_1d = None
        self.data_4h = None
        self.data_1h = None
        self.data_15m = None
        self.data_5m = None
        self.data_1m = None

    def load_data(self, ticker):
        self.data_1w = pd.read_csv(f"candle_data/{ticker}_1w_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_1d = pd.read_csv(f"candle_data/{ticker}_1d_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_4h = pd.read_csv(f"candle_data/{ticker}_4h_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_1h = pd.read_csv(f"candle_data/{ticker}_1h_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_15m = pd.read_csv(f"candle_data/{ticker}_15m_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_5m = pd.read_csv(f"candle_data/{ticker}_5m_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_1m = load_data_1m(ticker)
        print("[Data]: load data completed")

    def load_data_with_normalization(self, ticker):
        self.load_data(ticker)
        self.normalization()
        # print(self.data_1w, self.data_1d, self.data_4h, self.data_1h,self.data_15m, self.data_5m, self.data_1m)
        

    def normalization(self):
        start = time.time()
        self.min_max()
        print("Normalization time :", time.time() - start)
    
    def min_max(self):
        for col in scale_cols:
            self.data_1w[col] = min_max_normalize(self.data_1w[col])
            self.data_1d[col] = min_max_normalize(self.data_1d[col])
            self.data_4h[col] = min_max_normalize(self.data_4h[col])
            self.data_1h[col] = min_max_normalize(self.data_1h[col])
            self.data_15m[col] = min_max_normalize(self.data_15m[col])
            self.data_5m[col] = min_max_normalize(self.data_5m[col])
            self.data_1m[col] = min_max_normalize(self.data_1m[col])

        self.data_1w[cost_cols] = scaler.fit_transform(self.data_1w[cost_cols])
        self.data_1d[cost_cols] = scaler.fit_transform(self.data_1d[cost_cols])
        self.data_4h[cost_cols] = scaler.fit_transform(self.data_4h[cost_cols])
        self.data_1h[cost_cols] = scaler.fit_transform(self.data_1h[cost_cols])
        self.data_15m[cost_cols] = scaler.fit_transform(self.data_15m[cost_cols])
        self.data_5m[cost_cols] = scaler.fit_transform(self.data_5m[cost_cols])
        self.data_1m[cost_cols] = scaler.fit_transform(self.data_1m[cost_cols])

        self.data_1w[volume_cols] = scaler.fit_transform(self.data_1w[volume_cols])
        self.data_1d[volume_cols] = scaler.fit_transform(self.data_1d[volume_cols])
        self.data_4h[volume_cols] = scaler.fit_transform(self.data_4h[volume_cols])
        self.data_1h[volume_cols] = scaler.fit_transform(self.data_1h[volume_cols])
        self.data_15m[volume_cols] = scaler.fit_transform(self.data_15m[volume_cols])
        self.data_5m[volume_cols] = scaler.fit_transform(self.data_5m[volume_cols])
        self.data_1m[volume_cols] = scaler.fit_transform(self.data_1m[volume_cols])



    



    def get_datas(self):
        # return 값은 list
        return [self.data_1w, self.data_1d, self.data_4h,
                self.data_1h, self.data_15m, self.data_5m, self.data_1m]
    def get_datas_len(self):
        return self.data_1w.shape[1] + self.data_1d.shape[1] + self.data_4h.shape[1] + self.data_1h.shape[1] + self.data_15m.shape[1] + self.data_5m.shape[1] + self.data_1m.shape[1]

    def get_datas_shape(self):
        return len(self.data_1w.columns)

    def print_data_columns(self):
        print(self.data_1w.columns)
        print(self.data_1d.columns)
        print(self.data_4h.columns)
        print(self.data_1h.columns)
        print(self.data_15m.columns)
        print(self.data_5m.columns)
        print(self.data_1m.columns)


if __name__ == '__main__':
    data = Data()


