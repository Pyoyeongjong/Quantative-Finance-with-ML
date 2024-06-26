import gym
from gym import spaces
import numpy as np
import pandas as pd
from data import Data
import time

tickers = ["BTCUSDT","ETHUSDT", "BNBUSDT","SOLUSDT","XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT","DOTUSDT",
            "LINKUSDT", "TRXUSDT", "MATICUSDT","BCHUSDT", "ICPUSDT",
            "NEARUSDT", "UNIUSDT", "APTUSDT", "LTCUSDT", "STXUSDT",
            "FILUSDT", "THETAUSDT", "NEOUSDT", "FLOWUSDT", "XTZUSDT"]
ticker = ["BTCUSDT"] # for test

timezone = ["1w", "1d", "4h", "1h", "15m", "5m", "1m"]

# low_arr = np.array([0, 0.08 * 0.01, 0])
# high_arr = np.array([0.9, 1000, 1])
# 31536000 = 1 year / (sec) max position holding time
MAX_POS_HOLD_TIME= 31536000
BALANCE = 10000
TRANS_FEE = 0.04 * 0.01

last_row_index = [0, 0, 0, 0, 0, 0, 0]

def init_lri():
    for i in range(len(last_row_index)):
        last_row_index[i] = 0

class BitcoinTradingEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, low_arr, high_arr):
        self.curr = 0
        self.curr_ticker = 0
        self.done = False
        self.balance = BALANCE
        self.datas = Data()
        self.datas.load_data_initial(tickers[self.curr_ticker])
        self.budget = 10000
        super(BitcoinTradingEnv, self).__init__()
        # 0 = 산다, 1= 판다, 2 = 관망
        # Box = (손절가, 익절가, 들어가는 양)
        self.action_space = spaces.Tuple(spaces=(spaces.Discrete(n=3),
                                                 spaces.Box(low=low_arr, high=high_arr, dtype=np.float64)))
        print(self.datas.get_datas_len())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.datas.get_datas_len(), ), dtype=np.float64) # (data.shape[1], ) = (열, 1)

        print("[BitcoinTradingEnv]: Env init OK")

    def cal_reward(self, percent):
        if percent >= 0:
            # return percent
            return percent
        else:
            return - (1 / (1 + percent) - 1)
            #return - ((1 + abs(percent)) ** 1.1 - 1) * 100


    def next_observation(self):

        drop_list = ['open', 'high', 'low', 'close', 'volume']

        start = time.time()
        datas = self.datas.get_obs_datas()
        curr_time = self.datas.data_1m.loc[self.curr, 'time']
        print("cur=",self.curr, " curr_time=", curr_time)

        rows_list = []
        data_loc = 0
        for data in datas:

            # 04.10 데이터의 일봉, 시간봉이 내가 원하는 대로 나오는가? 를 테스트
            # 아 신규코인은 전 일봉 시간봉 데이터가 없으니까 못가져올 수도 있겠다
            # data.index.get_loc(curr_asof) - 1 가 음수면 curr을 올리고 next_obs를 한번 더 하는 로직을 짜는 걸 예상할 수 있음. ok!

            # 04.11 노가다로 최적화해야겠는데.. 노가다가 아닌 방법이 있을 수도 있다..!

            # print(last_row_index[data_loc])

            filtered_data = data.loc[(data.index >= int(last_row_index[data_loc])) & (data['time'] <= curr_time)]

            if not filtered_data.empty:
                row = filtered_data.iloc[-1]
            else:
                print("No rows match the condition.")
                return None  

            if int(row.name) >  1: 
                last_row_index[data_loc] = int(row.name) - 1
            else:
                last_row_index[data_loc] = 0

            # print(datetime.fromtimestamp(int(row['time'])/1000))
            # print(last_row_index[data_loc])
            data_loc += 1
            # print(row)
            # 04.10 만약 결측지가 있다면 빼는 코드
            # 근데 결측지가 왜 있지???

            # self.datas.print_data_columns()
            rows_list.append(row)

        # print("[NEXT_OBS] finish_time=",(time.time() - start))
        rows = np.concatenate(rows_list)
        print("next_observation time=",time.time() - start)
        ## 04.20 nan해결하러가자
        if np.isnan(rows).any():
            print("There is nan data.")
            time.sleep(100000)
            self.curr += 1
            self.next_observation()

        return rows
    
    def wait_until_end(self, action, sl, tp):

        start = time.time()

        # return 값
        # 0 = Win, 1 = Lose, 2 = Can't Explain

        num_rows = len(self.datas.data_1m)
        count = 0
        self.curr += 1

        lose = False
        win = False
        # print("[Waint until end] sl=",sl," tp=",tp)

        while count < MAX_POS_HOLD_TIME:
            # ticker의 1m을 다 돌았을 때 판단할 수 없음.
            if self.curr >= num_rows:
                print("curr > num_rows")
                return 2, count

            low = pd.to_numeric(self.datas.data_1m.loc[self.curr, "low"]) # 
            high = pd.to_numeric(self.datas.data_1m.loc[self.curr, "high"])
            # print("low=",low," high=",high)
            
            if action == 0:
                if low <= sl :
                    lose = True
                if tp <= high:
                    win = True
            elif action == 1:
                if sl <= high :
                    lose = True
                if low <= tp :
                    win = True

            if lose and not win:
                # print("wait until time=",time.time() - start)
                return 1, count
            elif win and not lose:
                # print("wait until time=",time.time() - start)
                return 0, count
            elif lose and win:
                print("lose == win")
                return 2, count

            count += 1
            self.curr += 1
            lose = False
            win = False

        print("count > MAX_HOD_TIME")
        return 2, count

    def take_action(self, ddqn_act, ddpg_act):
        # print("[Take action]: act= ",ddqn_act," ddpg=",ddpg_act)
        start = time.time()
        curr_price = pd.to_numeric(self.datas.data_1m.loc[self.curr, "close"])
        action_type = ddqn_act # ddqn_act의 첫번쨰 값은 진짜 뭘까:????
        slp = ddpg_act[0][0] # stop loss percent 0.01 = 1%
        tpp = ddpg_act[0][1] # take profit percent
        amount = ddpg_act[0][2]
        
        if action_type == 2:
            print("[Take action]: act= 2")
            # 04.14 이거 간과해선 안될 듯
            return 0
        
        if np.isnan(slp) or np.isnan(tpp) or np.isnan(amount):
            print("ddpg is NaN")
            return None

        if action_type == 0:
            stop_loss = curr_price * (1 - slp)
            take_profit = curr_price * (1 + tpp)
        elif action_type == 1:
            stop_loss = curr_price * (1 + slp)
            take_profit = curr_price * (1 - tpp)

        # print("curr_pric=",curr_price, " stop_loss=",stop_loss," take_profit=", take_profit)
    
        # 0 = win, 1 = lose, 2 = Can't explain
        result, count = self.wait_until_end(action_type, stop_loss, take_profit)

        if result == 0:
            percent = (tpp - 2 * TRANS_FEE) * amount
        elif result == 1:
            percent = -(slp + 2 * TRANS_FEE) * amount
        else:
            percent = None

        if percent is None:
            reward = None
            print("[Take action]: reward is None")

        else:
            reward = self.cal_reward(percent)
            self.budget = self.budget * (1 + percent)

            # print("take action time=",time.time() - start)
                    # reward는 시간도 고려해야함!
            # if count > 1:
            #     reward = reward / ( count**(0.5) )
    
            print(f"[Take action]: act= {ddqn_act} slp= {slp:.4f}, tpp= {tpp:.4f}, amt= {amount}, result= {result} ",
                    f" reward= {reward:.5f}, budget= {self.budget:.3f}")

        return reward


    def soft_done(self):
        if self.curr >= len(self.datas.data_1m):
            return True
        else:
            return False

    def step(self, action):

        start = time.time()

        # Implement step here
        # action을 두개로 나눠서 받는다.
        ddqn_act = action[0]
        ddpg_act = action[1]

        # action에 따른 차트에 영향이 매우 미미하다 가정. 다음 observation 만 도출
        reward = self.take_action(ddqn_act, ddpg_act)

        self.curr += 1

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
        init_lri()
        return self.next_observation()

    def soft_reset(self):

        if self.curr_ticker >= len(tickers):
            self.done = True
            return
        
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
            init_lri()

    def render(self, mode='console'):
        if mode == 'console':
            print("Custom environment render")
        else:
            return

    def close(self):
        pass # 아무것도 하지 않는다.


# 험난한 여정이 될 것 같다..

if __name__ == '__main__':
    data = Data()
    for ticker in tickers:
        print(f"\n\n{ticker}")
        data.load_data_with_normalization(ticker)