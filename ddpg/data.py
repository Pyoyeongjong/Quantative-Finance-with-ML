import pandas as pd
import numpy as np
import time
import os
import bitcoinA2Cenv
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max Scaling
scaler_minmax = MinMaxScaler()

# Z-score Standardization
scaler_standard = StandardScaler()



## z정규화해야할 cols
z_cols = ['openp', 'highp', 'lowp', 'closep',
          'sma5p', 'sma10p', 'sma20p', 'sma40p', 'sma60p','sma90p','sma120p', 'ema5p', 'ema20p','ema60p','ema120p','volp',
          'upperbandp', 'lowerbandp', 'atr', 'cci', 'adx']

z_cols_old = ['openp', 'highp', 'lowp', 'closep',
          'sma7p',  'sma20p',  'sma60p','sma120p','volp',
          'upperbandp', 'lowerbandp', 'atr', 'cci', 'adx']
## z정규화할 cols 
# z_cols2 = ['atr', 'cci', 'adx']

## min_max cols
min_max_cols = ['rsi']

# 04.20 volp는 원래 없어야 해
drop_list = ['open', 'high', 'low', 'close', 'volume', 'sma10p', 'sma40p', 'sma90p', 'ema5p', 'ema20p','ema60p','ema120p']

year = 2018

def load_data_1m(ticker):
    file_list = []
    for y in range(year, 2024):
        for i in range(0, 4):
            file_list.append(f"candle_datas/{ticker}_1m_{y}_{i}_sub.csv")
    df = [pd.read_csv(file).drop(columns='Unnamed: 0').dropna() for file in file_list if os.path.exists(file)]
    df_combine = pd.concat(df, ignore_index=True)

    return df_combine

def load_data_5m(ticker):
        
    file_list = []
    for y in range(year, 2024):
        file_list.append(f"candle_datas/{ticker}_5m_{y}_sub.csv")
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

        self.data_1w_obs = None
        self.data_1d_obs = None
        self.data_4h_obs = None
        self.data_1h_obs = None
        self.data_15m_obs = None
        self.data_5m_obs = None
        self.data_1m_obs = None

        self.data_1w_mv = None
        self.data_1d_mv = None
        self.data_4h_mv = None
        self.data_1h_mv = None

        self.data_attributes = [
            'data_1w', 'data_1d', 'data_4h', 
            'data_1h'# data_15m'#  'data_5m' #'data_1m'
        ]

    def load_data(self, ticker):
        start = time.time()
        self.data_1w = pd.read_csv(f"candle_datas/{ticker}_1w_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_1d = pd.read_csv(f"candle_datas/{ticker}_1d_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_4h = pd.read_csv(f"candle_datas/{ticker}_4h_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_1h = pd.read_csv(f"candle_datas/{ticker}_1h_sub.csv").drop(columns='Unnamed: 0').dropna()
        # self.data_15m = pd.read_csv(f"candle_datas/{ticker}_15m_sub.csv").drop(columns='Unnamed: 0').dropna()
        # self.data_5m = load_data_5m(ticker)
        # self.data_1m = load_data_1m(ticker)
        print("[Data]: load data completed. time=",time.time()-start)

    def make_mean_var_table(self, ticker): # 0이 mean, 1이 variance

        self.load_data(ticker)

        for attr in self.data_attributes:
            
            # 원본을 가져오는거다!!
            data = getattr(self, attr)
            # inf값 없애기
            data.replace([np.inf, -np.inf], 0, inplace=True)

            mean_list = []
            var_list = []

            for col in z_cols:
                mean = data[col].mean()
                var = data[col].var()

                print(f"{attr} {col} : ", mean, var)

                mean_list.append(mean)
                var_list.append(var)

            mean_var_list = [mean_list, var_list]
            df = pd.DataFrame(mean_var_list, columns=z_cols)
            df.to_csv(f"mv_table/{ticker}_{attr}_mv_table.csv")


    def load_test_data(self, ticker):
        start = time.time()
        self.data_1w = pd.read_csv(f"candle_datas_test/{ticker}_1w_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_1d = pd.read_csv(f"candle_datas_test/{ticker}_1d_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_4h = pd.read_csv(f"candle_datas_test/{ticker}_4h_sub.csv").drop(columns='Unnamed: 0').dropna()
        self.data_1h = pd.read_csv(f"candle_datas_test/{ticker}_1h_sub.csv").drop(columns='Unnamed: 0').dropna()
        print("[Data]: load test data completed. time=",time.time()-start)

    def load_test(self, ticker):
        self.load_test_data(ticker)
        self.normalization()
        self.load_obs_data()

    def load_data_with_normalization(self, ticker):
        self.load_data(ticker)
        self.normalization()
        self.load_obs_data()
        # print(self.data_1w, self.data_1d, self.data_4h, self.data_1h,self.data_15m, self.data_5m, self.data_1m)

    def load_test_with_mv(self, ticker):
        self.load_test_data(ticker)
        self.download_mv(ticker)
        self.z_norm_with_mv()
        self.mm_norm()
        self.load_obs_data()
        print("OK")

  
    
    def load_data_initial(self,ticker):
        self.data_1w = pd.read_csv(f"candle_datas/{ticker}_1w_sub.csv", nrows=5).drop(columns='Unnamed: 0').dropna()
        self.data_1d = pd.read_csv(f"candle_datas/{ticker}_1d_sub.csv", nrows=35).drop(columns='Unnamed: 0').dropna()
        self.data_4h = pd.read_csv(f"candle_datas/{ticker}_4h_sub.csv", nrows=200).drop(columns='Unnamed: 0').dropna()
        self.data_1h = pd.read_csv(f"candle_datas/{ticker}_1h_sub.csv", nrows=800).drop(columns='Unnamed: 0').dropna()
        # self.data_15m = pd.read_csv(f"candle_datas/{ticker}_15m_sub.csv", nrows=1000).drop(columns='Unnamed: 0').dropna()
        # self.data_5m = pd.read_csv(f"candle_datas/{ticker}_5m_2023_sub.csv", nrows=1000).drop(columns='Unnamed: 0').dropna()
        # self.data_1m = pd.read_csv(f"candle_datas/{ticker}_1m_2023_3_sub.csv", nrows=1000).drop(columns='Unnamed: 0').dropna()
        self.load_obs_data()


    def z_norm(self):
        start = time.time()
        for attr in self.data_attributes:
            # 원본을 가져오는거다!!
            data = getattr(self, attr)

            # inf값 없애기
            data.replace([np.inf, -np.inf], 0, inplace=True)
            for row in z_cols:
                data[row] = scaler_standard.fit_transform(data[[row]])

        print("[Data]: z_norm completed. time=",time.time()-start)

    def download_mv(self, ticker):
        start = time.time()
        self.data_1w_mv = pd.read_csv(f"mv_table/{ticker}_data_1w_mv_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_1d_mv = pd.read_csv(f"mv_table/{ticker}_data_1d_mv_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_4h_mv = pd.read_csv(f"mv_table/{ticker}_data_4h_mv_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_1h_mv = pd.read_csv(f"mv_table/{ticker}_data_1h_mv_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)

    def z_norm_with_mv(self):

        for attr in self.data_attributes:
            # 원본을 가져오는거다!!
            data = getattr(self, attr)
            data_mv = getattr(self, attr+"_mv")

            # inf값 없애기
            data.replace([np.inf, -np.inf], 0, inplace=True)
            for col in z_cols:
                scaler_standard.mean_ = np.array([data_mv[col][0]])
                scaler_standard.scale_ = np.array([math.sqrt(data_mv[col][1])])
                data[col] = scaler_standard.transform(data[[col]])


    def mm_norm(self):
        start = time.time()
        for attr in self.data_attributes:
            # 원본을 가져오는거다!!
            data = getattr(self, attr)
            for row in min_max_cols:
                data[row] = scaler_minmax.fit_transform(data[[row]])
        print("[Data]: mm_norm completed. time=",time.time()-start)
                
    def normalization(self):
        start = time.time()
        # self.min_max()
        self.z_norm()
        self.mm_norm()

    def load_obs_data(self):
        self.data_1w_obs = self.data_1w.drop(columns=drop_list)
        self.data_1d_obs = self.data_1d.drop(columns=drop_list)
        self.data_4h_obs = self.data_4h.drop(columns=drop_list)
        self.data_1h_obs = self.data_1h.drop(columns=drop_list)
        # self.data_15m_obs = self.data_15m.drop(columns=drop_list)
        # self.data_5m_obs = self.data_5m.drop(columns=drop_list)
        # self.data_1m_obs = self.data_1m.drop(columns=drop_list)

    def get_datas(self):
        # return 값은 list
        return [self.data_1w, self.data_1d, self.data_4h,
                self.data_1h]
        # self.data_15m # self.data_1m] , self.data_5m,
    
    def get_obs_datas(self):
        # return 값은 list
        return [self.data_1w_obs, self.data_1d_obs, self.data_4h_obs,
                self.data_1h_obs]
        # self.data_1m_obs] , self.data_15m_obs
    
    def get_datas_len(self):
        return self.data_1w.shape[1] + self.data_1d.shape[1] + self.data_4h.shape[1] + self.data_1h.shape[1] - len(drop_list) * len(self.data_attributes) # + self.data_1m.shape[1]

    def get_datas_shape(self):
        return len(self.data_1w.columns)

    def print_data_columns(self):
        print(self.data_1w.columns)
        print(self.data_1d.columns)
        print(self.data_4h.columns)
        print(self.data_1h.columns)
        # print(self.data_15m.columns)
        # print(self.data_5m.columns)
        # print(self.data_1m.columns)


if __name__ == '__main__':
    data = Data()
    data.load_test_with_mv("BTCUSDT")





