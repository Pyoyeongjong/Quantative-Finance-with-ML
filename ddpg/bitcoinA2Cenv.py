import gym
from gym import spaces
import numpy as np
import pandas as pd
from data import Data
import time


## Data Preparation

# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df)

tickers = ["BTCUSDT","ETHUSDT", "BNBUSDT","SOLUSDT","XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT","DOTUSDT",
            "LINKUSDT", "TRXUSDT", "MATICUSDT","BCHUSDT", "ICPUSDT",
            "NEARUSDT", "UNIUSDT", "APTUSDT", "LTCUSDT", "STXUSDT",
            "FILUSDT", "THETAUSDT", "NEOUSDT", "FLOWUSDT", "XTZUSDT"]
ticker = ["BTCUSDT"] # for test

timezone = ["1w", "1d", "4h", "1h", "15m", "5m", "1m"]

from gym.envs.registration import register

BALANCE = 10000
TRANS_FEE = 0.04 * 0.01

lri = [-1, -1, -1, -1, -1, -1, -1] # last row index

def init_lri():
    for i in range(len(lri)):
        lri[i] = -1 # -1이 초기값

class BitcoinTradingEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self):
        self.curr = 0 # 1분봉 기준 현재 행의 위치
        self.curr_ticker = 24 # tickers 리스트에서 현재 사용 중인 index 
        self.done = False
        self.balance = BALANCE
        self.datas = Data()
        self.datas.load_data_initial(tickers[self.curr_ticker])
        self.budget = 10000
        super(BitcoinTradingEnv, self).__init__()
        # 0 = 산다, 1= 판다, 2 = 관망
        self.action_space = spaces.Discrete(3) # 롱, 숏, 관망
        self.long_action_space = spaces.Discrete(2) # 홀딩, 정리
        self.short_action_space = spaces.Discrete(2) # 홀딩, 정리
        print(self.datas.get_datas_len())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.datas.get_datas_len()-7, ), dtype=np.float64) # (data.shape[1], ) = (열, 1)

        print("[BitcoinTradingEnv]: Env init OK")

    def cal_reward(self, percent):
        if percent >= 0:
            return percent * 100
        else:
            return - (1 / (1 + percent) - 1) * 100

    
    def ticker_is_done(self):
        if self.curr >= len(self.datas.data_1m):
            return True
        else:
            return False
        
    def get_curr_ohlcv(self, ohlcv):
        return self.datas.data_1m.loc[self.curr, ohlcv]
        
    # curr의 다음 행을 가져오는 함수.
    def get_next_row_obs(self):

        start = time.time()
        timestamp = [604800, 86400, 14400, 3600, 900, 300, 60] * 1000

        ohlcv_list = ['open', 'high', 'low', 'close', 'volume'] # OHLCV
        self.curr += 1

        datas = self.datas.get_obs_datas()
        curr_time = self.datas.data_1m.loc[self.curr, 'time']
        rows_list = []
        data_loc = 0

        for data in datas:

            # 1m 에 대해선 안해도 됨.
            if data_loc >= 6:
                lri[data_loc] = self.curr
            else:
                # 만약 초기값 상태라면, 1m time보다 작은 가장 큰 행 위치를 lri에 기록
                if int(lri[data_loc]) < 0:
                    filtered_data = data.loc[data['time'] <= curr_time]
                    if not filtered_data.empty:
                        lri[data_loc] = filtered_data.index[-1]

                # row보다 1m time + 시간프레임이 크다면 index += 1
                if lri[data_loc] < len(data) - 1:
                    compare_row = data.loc[lri[data_loc]]
                    if compare_row['time'] <= curr_time + timestamp[data_loc]:
                        lri[data_loc] += 1

            row = data.loc[lri[data_loc]]
            row = row.drop("time")
            rows_list.append(row)
            data_loc += 1

        rows = np.concatenate(rows_list)
        
        if np.isnan(rows).any():
            print("There is nan data.")
            time.sleep(10)
            self.get_next_row_obs()

        # 만약 마지막이라면?
        if self.ticker_is_done():
            self.done = True
            return None
        
        return self.get_curr_ohlcv(ohlcv_list), rows
        
    def cal_percent(self, before, after):
        return ((after - before) / before - TRANS_FEE)
        
        
    def long_step(self, action, position):

        ohlcv, obs = self.get_next_row_obs()

        if obs.any() is None: # 판단할 수 없음.
            reward = None
            done = True
        
        # action이 홀딩(0) 이면
        # obs = 다음 행 reward = 0, done = False, info = 대충 없음.
        if action == 0:
            done = False
            reward = 0

        # action이 정리(1) 이면
        # obs = 다음 행, reward = 이득, done = True, info = 대충 없음.
        if action == 1:
            done = True
            percent = self.cal_percent(position, ohlcv['close'])
            reward = self.cal_reward(percent)
        
        info = [ohlcv['close']]

        return obs, reward, done, info

    def short_step(self, action, position):

        ohlcv, obs = self.get_next_row_obs()

        if obs is None: # 판단할 수 없음.
            reward = None
            done = True
            info = ['sorry']
        
        # action이 홀딩(0) 이면
        # obs = 다음 행 reward = 0, done = False, info = 대충 없음.
        if action == 0:
            done = False
            reward = 0

        # action이 정리(1) 이면
        # obs = 다음 행, reward = 이득, done = True, info = 대충 없음.
        if action == 1:
            done = True
            percent = -self.cal_percent(position, ohlcv['close'])
            reward = self.cal_reward(percent)
        
        info = [ohlcv['close']]

        return obs, reward, done, info

    def reset(self):

        self.curr = 0
        if self.curr_ticker >= len(tickers) - 1:
            self.curr_ticker = 0
        else:
            self.curr_ticker += 1

        print("[Env]: Reset. ticker = ",tickers[self.curr_ticker])
        # Implement reset here

        self.budget = 10000
        self.done = False
        init_lri()

        # self.datas.load_data_initial(tickers[self.curr_ticker])
        self.datas.load_data_with_normalization(tickers[self.curr_ticker])
        return self.get_next_row_obs()



def main():

    max_episode_num = 200
    agent = Train()

    agent.train(max_episode_num)

if __name__=='__main__':
    main()