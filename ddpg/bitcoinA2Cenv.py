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

class BitcoinTradingEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, low_arr, high_arr):
        self.curr = 0 # 1분봉 기준 현재 행의 위치
        self.curr_ticker = 0 # tickers 리스트에서 현재 사용 중인 index 
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.datas.get_datas_len(), ), dtype=np.float64) # (data.shape[1], ) = (열, 1)

        print("[BitcoinTradingEnv]: Env init OK")

    def cal_reward(self, percent):
        if percent >= 0:
            return percent
        else:
            return - (1 / (1 + percent) - 1)

    
    def ticker_is_done(self):
        if self.curr >= len(self.datas.data_1m):
            return True
        else:
            return False
        
    def get_next_row_obs(self):

        # 만약 마지막이라면?
        if self.ticker_is_done():
            return None
        
        # curr의 다음 행을 가져오는 함수.

        
        
    def long_step(self, action):

        # action이 홀딩(0) 이면
        # obs = 다음 행 reward = 0, done = False, info = 대충 없음.

        obs = self.get_next_row_obs()
        if obs is None: # 판단할 수 없음.
            reward = None
            done = True
        
        if action == 0:
            done = False

        # action이 정리(1) 이면
        # obs = 다음 행, reward = 이득, done = True, info = 대충 없음.
        if action == 1:
            done = True


        def cal_long_reward(action):
            if action == 0:
                return 0
            else:
                
                return self.cal_reward(percent)
        
        info = "Long"

        if reward is not None:
            reward = cal_long_reward()

        return obs, reward, done, info

    def short_step(self, action):

        # action이 홀딩(0) 이면
        # obs = 다음 행 reward = 0, done = done, info = 대충 없음.


        # action이 정리(1) 이면
        # obs = 다음 행, reward = 결과, done = done, info = 대충 없음.
        

        def cal_short_reward():
            return 0
        
        reward = cal_short_reward()
        info = "Long"

        return obs, reward, self.done, info


    # 이건 env 가 아닌 agent 에 있어야 할 함수! 편의상 여기서 먼저 만듬.
    def action_step(self, action):

        # action 이 매수(0)이면
        # long_step을 통해 진행
        if action == 0:
            state = self.get_next_row_obs()

            while(1): # long action이 종료될 때 까지
                long_act = self.DDQN.act(state, self.long_action_space.shape[0])
                l_next_state, l_reward, l_done, _ = self.long_step(long_act)

                if l_done:
                    break # long 거래가 끝났으므로 빠져나간다.

            return obs, reward, done, info





        # action 이 매도(1)이면
        # short_step을 통해 진행
        if action == 1:
            state = self.get_next_row_obs()
            while(1): # long action이 종료될 때 까지
                short_act = self.DDQN.act(state, self.short_action_space.shape[0])
                s_next_state, s_reward, s_done, _ = self.long_step(long_act)

                if l_done:
                    break # long 거래가 끝났으므로 빠져나간다.

            reward = s_reward
            done = self.done

            return obs, reward, done, info

        # 관망(2)이면
        # next_obs 그냥 다음 행 주기.

        if self.soft_done():
            self.soft_reset()

        obs = self.next_observation()
        if obs is None:
            self.soft_reset()
            obs = self.next_observation()

        done = self.done
        info = {"reward": reward} # for debug
        # print("[STEP]: step done, final_reward=", reward)
        # print("step time=",time.time() - start)
        return obs, reward, done, info

    def reset(self):

        print("[Env]: hard reset. first ticker = ",tickers[0])
        # Implement reset here
        self.curr = 0
        self.curr_ticker = 0
        self.budget = 10000
        self.done = False

        # self.datas.load_data_with_normalization(tickers[self.curr_ticker])
        self.datas.load_data_with_normalization(tickers[self.curr_ticker])
        return self.next_observation()

    def soft_reset(self):
        
        print("curr_ticker=",self.curr_ticker)
        print(len(tickers))

        with open('./save_weights/train_budget.txt', 'a') as f:
            f.write(f"{tickers[self.curr_ticker]} : {self.budget}\n")

        self.budget = 10000
        self.curr = 0
        self.curr_ticker += 1
        if self.curr_ticker >= len(tickers):
            self.done = True
            return
        
        else:
            print("[Env]: soft reset to ", tickers[self.curr_ticker])
            # self.datas.load_data_with_normalization(tickers[self.curr_ticker])
            self.datas.load_data_with_normalization(tickers[self.curr_ticker])



if __name__ == '__main__':
    data = Data()
    for ticker in tickers:
        print(f"\n\n{ticker}")
        data.load_data_with_normalization(ticker)