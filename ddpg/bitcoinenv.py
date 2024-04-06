import gym
from gym import spaces
import numpy as np
from data import Data

from sklearn.preprocessing import MinMaxScaler

## Data Preparation

# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df)

tickers = ["BTCUSDT","ETHUSDT", "BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT", "ADAUSDT", "AVAXUSDT",
           "SHIBUSDT","DOTUSDT", "LINKUSDT", "TRXUSDT", "MATICUSDT",
           "BCHUSDT", "ICPUSDT", "NEARUSDT", "UNIUSDT", "APTUSDT",
            "LTCUSDT", "STXUSDT"]

timezone = ["1w", "1d", "4h", "1h", "15m", "5m", "1m"]
year = [2021, 2022, 2023]

from gym.envs.registration import register

low_arr = np.array([0, 0.08 * 0.01, 0])
high_arr = np.array([0.9, 1000, 1])
# 31536000 = 1 year / (sec) max position holding time
MAX_POS_HOLD_TIME= 31536000
BALANCE = 10000
TRANS_FEE = 0.04 * 0.01


def cal_reward(percent):

    if percent >= 0:
        return percent
    else:
        return - (abs(percent) ** 1.2)


class BitcoinTradingEnv(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self):
        self.curr = 0
        self.curr_ticker = 0
        self.done = False
        self.balance = BALANCE
        self.datas = Data()
        self.datas.load_data(tickers[self.curr_ticker])
        super(BitcoinTradingEnv, self).__init__()
        # 0 = 산다, 1= 판다, 2 = 관망
        # Box = (손절가, 익절가, 들어가는 양)
        self.action_space = spaces.Tuple(spaces=(spaces.Discrete(n=3),
                                                 spaces.Box(low=low_arr, high=high_arr, dtype=np.float32)))
        print(self.datas.get_datas_len())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.datas.get_datas_len(), ), dtype=np.float32) # (data.shape[1], ) = (열, 1)


    def next_observation(self):

        datas = self.datas.get_datas()
        curr_time = self.datas.data_1m.loc[self.curr, 'time']
        rows = np.array([])

        for data in datas:
            curr_asof = data.index.asof(curr_time)
            row = data.iloc[data.index.get_loc(curr_asof) - 1]
            print(len(row))
            # self.datas.print_data_columns()
            rows = np.concatenate([rows, row])

        return np.array(rows)

    def wait_until_end(self, sl, tp):

        # return 값
        # 0 = Win, 1 = Lose, 2 = Can't Explain

        num_rows = len(self.datas.data_1m)
        count = 0
        self.curr += 1

        lose = False
        win = False

        while count < MAX_POS_HOLD_TIME:

            # ticker의 1m을 다 돌았을 때 판단할 수 없음.
            if self.curr >= num_rows:
                return 2

            low = self.datas.data_1m.loc[self.curr, "low"]
            high = self.datas.data_1m.loc[self.curr, "high"]

            if low <= sl <= high:
                lose = True
            if low <= tp <= high:
                win = True

            if lose and not win:
                return 1
            elif win and not lose:
                return 0

            count += 1
            self.curr += 1

        return 2

    def take_action(self, ddqn_act, ddpg_act):

        curr_price = self.datas.data_1m.loc[self.curr, "Close"]
        action_type = ddqn_act
        slp = ddpg_act[0] # stop loss percent
        tpp = ddpg_act[1] # take profit percent
        amount = ddpg_act[2]

        if action_type == 2:
            return 0

        if action_type == 0:
            stop_loss = curr_price * (1 - slp)
            take_profit = curr_price * (1 + tpp)

        elif action_type == 1:
            stop_loss = curr_price * (1 + slp)
            take_profit = curr_price * (1 - tpp)

        # 0 = win, 1 = lose, 2 = Can't explain
        result = self.wait_until_end(stop_loss, take_profit)

        if result == 0:
            percent = tpp * amount - 2 * TRANS_FEE
        elif result == 1:
            percent = (slp * amount + 2 * TRANS_FEE)
        else:
            percent = 0

        return percent


    def soft_done(self):
        if self.curr >= len(self.datas.data_1m):
            return True
        else:
            return False

    def step(self, action):
        # Implement step here
        # action을 두개로 나눠서 받는다.
        ddqn_act = action[0]
        ddpg_act = action[1]

        # action에 따른 차트에 영향이 매우 미미하다 가정. 다음 observation 만 도출
        percent = self.take_action(ddqn_act, ddpg_act)

        self.curr += 1

        if self.soft_done():
            self.soft_reset()

        obs = self.next_observation()
        reward = cal_reward(percent)

        done = self.done
        info = {"percent": percent} # for debug
        return obs, reward, done, info

    def reset(self):
        # Implement reset here
        self.curr = 0
        self.curr_ticker = 0
        self.datas.load_data(tickers[self.curr_ticker])
        return self.next_observation()

    def soft_reset(self):
        self.curr = 0
        self.curr_ticker += 1
        if self.curr_ticker >= len(tickers):
            self.done = True
            return
        else:
            self.datas.load_data(tickers[self.curr_ticker])

    def render(self, mode='console'):
        if mode == 'console':
            print("Custom environment render")
        else:
            return

    def close(self):
        pass # 아무것도 하지 않는다.


# 험난한 여정이 될 것 같다..

if __name__ == '__main__':
    print('hi')