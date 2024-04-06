import pandas as pd
import numpy as np

year = [2021, 2022, 2023]


def load_data_1m(ticker):
    file_list = []
    for y in year:
        file_list.append(f"candle_data/{ticker}_1m_{y}_0_sub.csv")
        file_list.append(f"candle_data/{ticker}_1m_{y}_1_sub.csv")
    df = [pd.read_csv(file) for file in file_list]
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
        self.data_1w = pd.read_csv(f"candle_data/{ticker}_1w_sub.csv")
        self.data_1d = pd.read_csv(f"candle_data/{ticker}_1d_sub.csv")
        self.data_4h = pd.read_csv(f"candle_data/{ticker}_4h_sub.csv")
        self.data_1h = pd.read_csv(f"candle_data/{ticker}_1h_sub.csv")
        self.data_15m = pd.read_csv(f"candle_data/{ticker}_15m_sub.csv")
        self.data_5m = pd.read_csv(f"candle_data/{ticker}_5m_sub.csv")
        self.data_1m = load_data_1m(ticker)
        print("load data completed")

    def get_datas(self):
        # return 값은 list
        return [self.data_1w, self.data_1d, self.data_4h,
                self.data_1h, self.data_15m, self.data_5m, self.data_1m]
    def get_datas_len(self):
        return len(self.data_1m) + len(self.data_1w) + len(self.data_5m) + len(self.data_1h) + len(self.data_4h) + len(self.data_15m) + len(self.data_1d)

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

