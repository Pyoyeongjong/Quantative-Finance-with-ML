import gym
from gym import spaces
import numpy as np
import pandas as pd
from data import Data
import time
from datetime import datetime

# 파라미터
TRANS_FEE = 0.04 * 0.01
HOLD_REWARD = 0.01

def get_long_sl(position):
    return position * 0.9
def get_short_sl(position):
    return position * 1.1

# Ticker
tickerss = ["BTCUSDT","ETHUSDT", "BNBUSDT","SOLUSDT","XRPUSDT"]
# , "BNBUSDT","SOLUSDT","XRPUSDT",
#            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT","DOTUSDT",
#             "LINKUSDT", "TRXUSDT", "MATICUSDT","BCHUSDT", "ICPUSDT",
#             "NEARUSDT", "UNIUSDT", "APTUSDT", "LTCUSDT", "STXUSDT",
#             "FILUSDT", "THETAUSDT", "NEOUSDT", "FLOWUSDT", "XTZUSDT"]
# for test
tickers= ["ETCUSDT"]

tickersa = ["BTCUSDT","ETHUSDT", "BNBUSDT","SOLUSDT","XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT","DOTUSDT",
            "LINKUSDT", "TRXUSDT", "MATICUSDT","BCHUSDT", "ICPUSDT",
            "NEARUSDT", "UNIUSDT", "APTUSDT", "LTCUSDT", "STXUSDT",
            "FILUSDT", "THETAUSDT", "NEOUSDT", "FLOWUSDT", "XTZUSDT"]

# Timezone
timezone = ["1w", "1d", "4h", "1h", "15m", "5m", "1m"]

# Last Row Index -> get_next_obs 함수 최적화를 위해
lri = [-1, -1, -1, -1, -1, -1, -1] # last row index

# lri 초기화
def init_lri():
    for i in range(len(lri)):
        lri[i] = -1 # -1이 초기값

class BitcoinTradingEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, time_steps):
        self.TRANS_FEE = 0.04 * 0.01
        self.curr = 0 # 1분봉 기준 현재 행의 위치
        self.curr_ticker = len(tickers)-1 # tickers 리스트에서 현재 사용 중인 index 
        self.done = False # ticker 교체를 해야 하는가?
        self.datas = Data()
        self.datas.load_data_initial(tickers[self.curr_ticker]) # test용
        self.budget = 10000 # 초기 자본금
        super(BitcoinTradingEnv, self).__init__()
        # 0 = 산다, 1= 판다, 2 = 관망
        self.action_space = spaces.Discrete(3) # 0 = 롱, 1 = 숏, 2 = 관망
        self.long_action_space = spaces.Discrete(2) # 0 = 홀딩, 1 = 정리
        self.short_action_space = spaces.Discrete(2) # 0 = 홀딩, 1 = 정리
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.datas.get_datas_len()-len(self.datas.data_attributes), ), dtype=np.float64) # (data.shape[1], ) = (열, 1)

        # LSTM 용 여러개 obs 저장용
        self.time_steps = time_steps # 0 이면 DNN 쓰는 거임.
        self.lstm_obs = []

        print("[BitcoinTradingEnv]: Env init OK")
        

    # 현재 ticker가 끝났는지 확인하는 함수
    def ticker_is_done(self):
        if self.curr >= len(self.datas.data_1h): #self.datas.data_1m
            self.done = True
            return True
        else:
            self.done = False
            return False
        
    # curr에 대응하는 1m의 ohlcv 제공. Return : Series
    def get_curr_ohlcv(self, ohlcv):
        return self.datas.data_1h.loc[self.curr, ohlcv] # 1m

       
    # curr의 다음 행을 가져오는 함수.
    def get_next_row_obs(self):

        self.curr += 1
        # 만약 다음 행이 없다면?
        if self.ticker_is_done():
            return None, None

        start = time.time()
        timestamp = [604800, 86400, 14400, 3600, 900, 300, 60] * 1000

        ohlcv_list = ['open', 'high', 'low', 'close', 'volume'] # OHLCV
        datas = self.datas.get_obs_datas()
        curr_time = self.datas.data_1h.loc[self.curr, 'time'] # 1m
        rows_list = []
        data_loc = 0

        for data in datas:
            # 1m 에 대해선 안해도 됨.
            if data_loc >= len(datas) - 1:
                lri[data_loc] = self.curr
            else:
                # 만약 초기값 상태라면, 1m time보다 작은 가장 큰 행 위치를 lri에 기록
                if int(lri[data_loc]) < 0:
                    filtered_data = data.loc[data['time'] <= curr_time]
                    if not filtered_data.empty:
                        lri[data_loc] = filtered_data.index[-1]
                    else: # 해당하는 row를 찾지 못함.
                        print(" row not find")
                        return None, None

                # row보다 1m time + 시간프레임이 크다면 index += 1
                if lri[data_loc] < len(data) - 1:
                    compare_row = data.loc[lri[data_loc]]
                    if curr_time - compare_row['time'] >= timestamp[data_loc] * 2:
                        lri[data_loc] += 1

            row = data.loc[lri[data_loc]-1]
            # Debug
            # print(datetime.fromtimestamp(int(row['time'])/1000))

            row = row.drop("time")
            rows_list.append(row)
            data_loc += 1

        rows = np.concatenate(rows_list)
        
        # row에 NaN이 있는지 검사
        if np.isnan(rows).any():
            print("There is nan data.")
            time.sleep(10)
            # 다음 obs 가져오기
            self.get_next_row_obs()
        
        # 1m ohlcv data
        ohlcv = self.get_curr_ohlcv(ohlcv_list)

        if self.time_steps == 0:
            # print("get_next_obs() exec time=",time.time() - start)
            return ohlcv, rows
        else: # LSTM 용 Next_OBS
            self.lstm_obs.append(rows)
            if len(self.lstm_obs) < self.time_steps:
                ohlcv, state = self.get_next_row_obs()
            if len(self.lstm_obs) > self.time_steps:
                self.lstm_obs.pop(0)
            # print(ohlcv, self.lstm_obs)
            # print("get_next_obs() exec time=",time.time() - start)
            return ohlcv, self.lstm_obs



        
    # 손익률 구하는 함수
    def cal_percent(self, before, after):
        return ((after - before) / before)
        
    # 여기서 return하는 done은 현재 포지션의 종료 여부다.
    def long_or_short_step(self, action, position, short):

        # 행동의 다음 봉
        ohlcv, obs = self.get_next_row_obs()
            
        # 판단할 수 없을 때
        if obs is None: 
            reward = None
            done = True
            return None, None, True, ["sorry"]
        # action이 홀딩(0) 이면
        # obs = 다음 행 reward = ??? 잘 생각해보자, done = False, info = 대충 없음.
        if action == 0:
            done = False
            # percent = self.cal_percent(position, ohlcv['close']) - self.cal_percent(position, ohlcv['open'])
            percent = 0
            if short: # 숏 추가지원금
                percent = percent
            # 홀딩 추가지원금
            percent += HOLD_REWARD
        # action이 정리(1) 이면
        # obs = 다음 행, reward = 포지션에 대한 이득??? 잘 생각해보자, done = True, info = 대충 없음.
        elif action == 1:
            done = True
            # percent = 0
            percent = self.cal_percent(position, ohlcv['open'])
            if short:
                percent = -percent
            percent = percent - TRANS_FEE

        long_sl = get_long_sl(position)
        short_sl = get_short_sl(position)
        # 강제 청산
        if ohlcv['low'] < long_sl and short == False: # 롱 청산 
            percent = self.cal_percent(position, long_sl)
            percent = percent - TRANS_FEE
            done = True
            info = [long_sl]
        elif ohlcv['high'] > short_sl and short == True: # 숏 청산 
            percent = - self.cal_percent(position, short_sl)
            percent = percent - TRANS_FEE
            done = True
            info = [short_sl]
        else:
            info = [ohlcv['open']]
        
        return obs, percent, done, info

    def reset(self, test):

        if self.curr_ticker >= len(tickers) - 1:
            self.curr_ticker = 0
        else:
            self.curr_ticker += 1

        print("[Env]: Reset. ticker = ",tickers[self.curr_ticker])
        # Implement reset here

        self.budget = 10000
        self.done = False
        init_lri()

        #self.datas.load_data_initial(tickers[self.curr_ticker])
        if test:
            self.datas.load_test(tickers[self.curr_ticker])
        else:
            self.datas.load_data_with_normalization(tickers[self.curr_ticker])
        # self.curr = self.datas.data_1h.shape[0] - 100
        self.curr = 0
        self.lstm_obs = []
        return self.get_next_row_obs()
    
    ### 여기부터는 Test함수

    def set_curr_to_timestamp(self, timestamp):
        while( self.datas.data_1h.loc[self.curr, 'time'] < timestamp ):
            # print(self.datas.data_1h.loc[self.curr, 'time'])
            self.curr += 1
        return


def main():
    return 0

if __name__=='__main__':
    main()