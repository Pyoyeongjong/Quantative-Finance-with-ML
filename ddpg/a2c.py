import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import layers, initializers
from replaybuffer import ReplayBuffer
from bitcoinA2Cenv import BitcoinTradingEnv
import bitcoinA2Cenv
from gym.spaces import Discrete, Space
import matplotlib.pyplot as plt
import time as TIME

# Test 날짜
TEST_TIMESTAMP = 1704067200 * 1000
XLM_TIMESTAMP = 1527814800 * 1000 # 2018/06.01

# 관망 리워드
STAY_REWARD = 0.000
STAY_DECAY = 1
INIT_DECAY = 1

# Reward 계산 함수
# percent = 원금에 대한 거래 손익 비율 ex) 0.05 = 5% 수익 
def cal_reward(percent):
    if percent >= 0:
        return percent
        # if percent >= 0.01:
        #     return (( percent * 100 ) ** 0.95) / 100
        # else:
        #     return percent
    else:
        return - (1 / (1 + percent) - 1)
        # return percent

# Ticker
# 이건 출력용이라 안바꿔도됌
tickers = bitcoinA2Cenv.tickers

class Actor:
    def __init__(self, time_steps, state_size, action_dim):
        self.time_steps = time_steps
        self.state_size = state_size # observation size
        self.action_dim = action_dim # 행동 사이즈
        if time_steps == 0: # time_steps가 0이면 DNN 사용
            self.model = self.build_model()
        else:
            self.model = self.build_LSTM_model()
        self.model.summary()

    def build_model(self):
        states = keras.Input(shape=(self.state_size,))
        out = layers.Dense(256, activation="relu",kernel_initializer=initializers.HeNormal())(states)
        out = layers.Dense(128, activation="relu",kernel_initializer=initializers.HeNormal())(out)
        actions = layers.Dense(self.action_dim, activation='softmax')(out)
        model = keras.Model(inputs=states, outputs=actions)
        
        return model
    
    def build_LSTM_model(self):
        states = keras.Input(shape=(self.time_steps, self.state_size))
        out = layers.LSTM(256, return_sequences=True, kernel_initializer=initializers.HeNormal())(states)
        out = layers.LSTM(128, return_sequences=False, kernel_initializer=initializers.HeNormal())(out)
        actions = layers.Dense(self.action_dim, activation='softmax')(out)
        model = keras.Model(inputs=states, outputs=actions)
        return model


class Critic:
    def __init__(self, time_steps, state_size):
        self.time_steps = time_steps
        self.state_size = state_size
        if time_steps == 0:
            self.model = self.build_model()
        else:
            self.model = self.build_LSTM_model()

        self.model.summary()

    def build_model(self):
        states = keras.Input(shape=(self.state_size,))
        out = layers.Dense(256, activation="relu",kernel_initializer=initializers.HeNormal())(states)
        out = layers.Dense(128, activation="relu",kernel_initializer=initializers.HeNormal())(out)
        q_values = layers.Dense(1, activation='linear')(out)
        model = keras.Model(inputs=states, outputs=q_values)
        return model
    
    def build_LSTM_model(self):
        states = keras.Input(shape=(self.time_steps, self.state_size))
        out = layers.LSTM(256, return_sequences=True, kernel_initializer=initializers.HeNormal())(states)
        out = layers.LSTM(128, return_sequences=False, kernel_initializer=initializers.HeNormal())(out)
        q_values = layers.Dense(1, activation='linear')(out)
        model = keras.Model(inputs=states, outputs=q_values)
        return model

class A2Cagent:

    def __init__(self, time_steps, state_dim, action_dim):
        # hyperparameters
        self.GAMMA = 0.9999
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(time_steps, self.state_dim, self.action_dim)
        self.critic = Critic(time_steps, self.state_dim)

        self.actor_opt = keras.optimizers.Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.critic_opt = keras.optimizers.Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.save_epi_reward = []

    # action 뱉기
    def get_action(self, state):
        state = np.array([state])
        # print("action_state=", state)
        action_probs = self.actor.model(state)
        # print(action_probs)
        # 합을 1로 만들기
        probs = action_probs[0]
        probs = np.clip(probs, a_min=0, a_max=None)  # 음수가 있을 경우 0으로 설정
        probs /= np.sum(probs)  # 합이 1이 되도록 정규화

        return probs

    def td_target(self, reward, v_value, next_v_value, done):
        if done:
            y_k = tf.constant([[reward]], dtype=tf.float32)
            advantage = y_k - v_value
        else:
            y_k = reward + self.GAMMA * next_v_value
            advantage = y_k - v_value
        return advantage, y_k
    
    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic.model(states, training=True)
            loss = tf.reduce_mean(tf.square(td_targets - td_hat))  # q - td_targets 을 통해 loss 계산

        grads = tape.gradient(loss, self.critic.model.trainable_variables)  # loss로 gradient 계산
        # Gradient Cliping
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.critic_opt.apply_gradients(zip(grads, self.critic.model.trainable_variables))  # critic 조정
    
    def actor_learn(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            probs = self.actor.model(states, training=True)  # states에 대한 action을 뽑아서
            action_probs = tf.reduce_sum(actions * probs, axis=1)
            log_probs = tf.math.log(action_probs + 1e-10)
            loss = -tf.reduce_mean(log_probs * advantages) # critic_q에 대한 loss계산

        grads = tape.gradient(loss, self.actor.model.trainable_variables)
        # Gradient Cliping
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.actor_opt.apply_gradients(zip(grads, self.actor.model.trainable_variables))


LONG_TYPE = 0
SHORT_TYPE = 1
ALL_TYPE = 2
class Train:

    def __init__(self, time_steps, agent_type):

        self.env = BitcoinTradingEnv(time_steps)

        self.agent_type = agent_type

        if agent_type == LONG_TYPE:
            self.TradeAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2) # 0 이 매수, 1 이 관망
            self.LongAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2)
        elif agent_type == SHORT_TYPE:
            self.TradeAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2)
            self.ShortAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2)
        else:
            self.TradeAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 3)
            self.LongAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2)
            self.ShortAgent = A2Cagent(time_steps, self.env.observation_space.shape[0], 2)

        self.BATCH_SIZE = 32
        self.BATCH_SIZE_LS = 128
        self.decay = INIT_DECAY
        # long ,short agent 용 batch들
        self.lstates, self.lactions, self.ltd_targets, self.ladvantages = [], [], [], []
        self.sstates, self.sactions, self.std_targets, self.sadvantages = [], [], [], []
        self.save_epi_reward = []
        print("Trainer OK")
        self.tradetrain = 0
        self.longtrain = 0
        self.shorttrain = 0

    def load_weights(self, path):
        self.TradeAgent.actor.model.load_weights(f"{path}/Trade_actor.weights.h5")
        self.TradeAgent.critic.model.load_weights(f"{path}/Trade_critic.weights.h5")
        if self.agent_type != SHORT_TYPE:
            self.LongAgent.actor.model.load_weights(f"{path}/Long_actor.weights.h5")
            self.LongAgent.critic.model.load_weights(f"{path}/Long_critic.weights.h5")
        if self.agent_type != LONG_TYPE:
            self.ShortAgent.actor.model.load_weights(f"{path}/Short_actor.weights.h5")
            self.ShortAgent.critic.model.load_weights(f"{path}/Short_critic.weights.h5")
        return

    def train(self, max_episode_num):

        print("Train start")

        # ep 한 번 = 한 ticker를 한바퀴 돌았을 때
        for ep in range(int(max_episode_num)):

            # initialize batch
            states, actions, td_targets, advantages = [], [], [], []

            
            start = TIME.time()

            # 에피소드 리셋
            time, episode_reward, done = 0, 0, False
            lowest_budget = self.env.budget
            highest_budget = self.env.budget

            # print("A:",TIME.time()-start)
            start = TIME.time()

            # Env 첫 state 가져오기
            _, state = self.env.reset(test=False)

            # print("B:",TIME.time()-start)
            start = TIME.time()

            while not done:
                
                ta_act = self.TradeAgent.get_action(state)
                act = np.random.choice(len(ta_act), p=ta_act)

                # All_TYPE일 때
                # if self.agent_type == ALL_TYPE:
                #     act = act
                # # LONG_TYPE
                # elif self.agent_type == LONG_TYPE:
                #     if act == 0:
                #         act = 0
                #     else:
                #         act = 2
                # # SHORT_TYPE
                # elif self.agent_type == SHORT_TYPE:
                #     if act == 0:
                #         act = 1
                #     else:
                #         act = 2

                # print("C:",TIME.time()-start)
                start = TIME.time()
                # 판단할 수 없을 때는 reward가 반드시 None이다.
                next_state, reward, done, info = self.action_step(act) 
                # print("D:",TIME.time()-start)
                start = TIME.time()
                if done == True: # Ticker가 종료되었다.
                    break
                if reward is None: # 종료되지 않고 판단할 수 없다 -> 다음 행을 검토한다.
                    _, state = self.env.get_next_row_obs()
                    continue

                if lowest_budget > self.env.budget:
                    lowest_budget = self.env.budget
                if highest_budget < self.env.budget:
                    highest_budget = self.env.budget

                # 여기서 done은 obs가 None일 떄(마지막일 떄) 만 True이다.
                # debug용
                if act < 2:
                    print(f"Time: {time}, Action: {act}, Reward: {reward*100:.2f}, Open: {info[0]:.5f}, Close: {info[1]:.5f}, Current: {info[2]:05}, Budget: {self.env.budget:.2f}")
                

                # TradeAgent학습
                v_value = self.TradeAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                next_v_value = self.TradeAgent.critic.model(tf.convert_to_tensor(np.array([next_state]), dtype=tf.float32))
                train_reward = reward
                # 사실 done이 True면 차트 특성상 가치평가를 할 수가 없긴 함
                advantage, y_i = self.TradeAgent.td_target(train_reward, v_value, next_v_value, done)
                # batch에 쌓기
                states.append(state)
                actions.append(ta_act)
                td_targets.append(y_i)
                advantages.append(advantage)
                # print("E:",TIME.time()-start)
                start = TIME.time()

                if len(states) == self.BATCH_SIZE:
                    # print("*****************************HI Action_learn********************************")
                    start = TIME.time()
                    # critic 학습
                    self.TradeAgent.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                           tf.convert_to_tensor(td_targets,dtype=tf.float32))
                    # actor 학습
                    self.TradeAgent.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                               tf.convert_to_tensor(actions, dtype=tf.float32),
                                           tf.convert_to_tensor(advantages,dtype=tf.float32))
                    self.tradetrain += 1
                    # batch 지우기
                    states, actions, td_targets, advantages = [], [], [], []
                    print("\nTradeAgent_learn exec time=",TIME.time() - start)

                episode_reward += reward
                state = next_state
                time += 1

            # while 밖
            # display each episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            ## 각 weight 저장하기
            # 저장 코드

            self.TradeAgent.actor.model.save_weights("./save_weights/Trade_actor.weights.h5")
            self.TradeAgent.critic.model.save_weights("./save_weights/Trade_critic.weights.h5")
            if self.agent_type != SHORT_TYPE:
                self.LongAgent.actor.model.save_weights("./save_weights/Long_actor.weights.h5")
                self.LongAgent.critic.model.save_weights("./save_weights/Long_critic.weights.h5")
            if self.agent_type != LONG_TYPE:
                self.ShortAgent.actor.model.save_weights("./save_weights/Short_actor.weights.h5")
                self.ShortAgent.critic.model.save_weights("./save_weights/Short_critic.weights.h5")
            print("[Save Weight]: ep",ep," save Completed")

            with open('./save_weights/train_reward.txt', 'a') as f:
                f.write(f"ep{ep} ticker = {tickers[self.env.curr_ticker]} reward = {episode_reward} budget = {self.env.budget} lowb = {lowest_budget} highb = {highest_budget}\n")
        
        # for 밖
        print("Train Finished")

    # obs 가 none이 불리면, agent가 reset을 진행.
    def action_step(self, action):

        done = False
        info = ["error", "sorry", self.env.curr]

        # action 이 매수(0)이면
        # long_step을 통해 진행
        if action == 0:
            self.decay = INIT_DECAY
            # self.env.curr -= 1 
            ohlcv, state = self.env.get_next_row_obs()
            if state is None:
                reward = None
                # ticker가 끝났는가?
                if self.env.ticker_is_done():
                    done = True # ticker끝
                else:
                    done = False
                info = ["obs is none"]
                return None, None, done, info # None, None, True, _
            
            reward = 0
            position = ohlcv['open'] # 임시
            close_position = None

            while(1): # long action이 종료될 때 까지
                long_act = self.LongAgent.get_action(state)
                act = np.random.choice(len(long_act), p=long_act)
                # obs가 None이거나, 거래거 끝났을 때 done = True
                l_next_state, l_reward, l_done, finish_info = self.env.long_or_short_step(act, position, False)

                # obs None 판단불가능 (ticker가 끝났을 때 밖에 없을 거 같은데...)
                if l_next_state is None:
                    reward = None
                    # ticker가 끝났는가?
                    if self.env.ticker_is_done():
                        done = True # ticker끝
                    else:
                        done = False
                    info = ["obs is none"]
                    return None, None, done, info # None, None, True, _
                
                # l_reward(percent) -> reward로 바꿔주기
                l_reward = cal_reward(l_reward)

                # LongAgent학습 (Main Agent와 빈도를 맞춰줘야 할 것 같다.)
                v_value = self.LongAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                next_v_value = self.LongAgent.critic.model(tf.convert_to_tensor(np.array([l_next_state]), dtype=tf.float32))
                train_reward = l_reward
                advantage, y_i = self.LongAgent.td_target(train_reward, v_value, next_v_value, l_done)
                # batch에 쌓기
                self.lstates.append(state)
                self.lactions.append(long_act)
                self.ltd_targets.append(y_i)
                self.ladvantages.append(advantage)
                # 학습하기
                if len(self.lstates) == self.BATCH_SIZE_LS:
                    if True: # 배우는 비율 맞추기 self.tradetrain * 10 > self.longtrain
                        start = TIME.time()
                        # print("*****************************HI Long_learn********************************")
                        # critic 학습
                        self.LongAgent.critic_learn(tf.convert_to_tensor(self.lstates, dtype=tf.float32),
                                            tf.convert_to_tensor(self.ltd_targets, dtype=tf.float32))
                        # actor 학습
                        self.LongAgent.actor_learn(tf.convert_to_tensor(self.lstates, dtype=tf.float32),
                                                tf.convert_to_tensor(self.lactions, dtype=tf.float32),
                                            tf.convert_to_tensor(self.ladvantages,dtype=tf.float32))
                        print("\nLongAgent_learn exec time=",TIME.time() - start)
                        self.longtrain += 1
                    # batch 지우기
                    self.lstates, self.lactions, self.ltd_targets, self.ladvantages = [], [], [], []
                # reward += l_reward
                state = l_next_state
                if l_done:
                    close_position = finish_info[0]
                    # 따로 reward를 계산해줘야 한다.
                    percent = self.env.cal_percent(position, close_position) - self.env.TRANS_FEE * 2
                    self.env.budget *= (1+percent)
                    reward = cal_reward(percent)
                    break # long 거래가 끝났으므로 빠져나간다.

            # budget에 넣기

            obs = state
            info = [position, close_position, self.env.curr]
            return obs, reward, done, info

        # action 이 매도(1)이면
        # short_step을 통해 진행
        elif action == 1:
            self.decay = INIT_DECAY
            ohlcv, state = self.env.get_next_row_obs()
            if state is None:
                reward = None
                # ticker가 끝났는가?
                if self.env.ticker_is_done():
                    done = True # ticker끝
                else:
                    done = False
                info = ["obs is none"]
                return None, None, done, info # None, None, True, _
            reward = 0
            position = ohlcv['open'] # 임시
            close_position = None

            while(1): # shont action이 종료될 때 까지
                short_act = self.ShortAgent.get_action(state)
                act = np.random.choice(len(short_act), p=short_act)
                # obs가 None이거나, 거래거 끝났을 때 done = True
                s_next_state, s_reward, s_done, finish_info = self.env.long_or_short_step(act, position, True)
                
                # obs None (ticker가 끝났을 때 밖에 없을 거 같은데...)
                if s_next_state is None:
                    reward = None
                    # ticker가 끝났는가?
                    if self.env.ticker_is_done():
                        done = True # ticker끝
                    else:
                        done = False
                    info = ["obs is none"]
                    return None, None, done, info # None, None, True, _
                
                # l_reward(percent) -> reward로 바꿔주기
                s_reward = cal_reward(s_reward)

                # ShortAgent학습 (Main Agent와 빈도를 맞춰줘야 할 것 같다.)
                v_value = self.ShortAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                next_v_value = self.ShortAgent.critic.model(tf.convert_to_tensor(np.array([s_next_state]), dtype=tf.float32))
                train_reward = s_reward
                advantage, y_i = self.ShortAgent.td_target(train_reward, v_value, next_v_value, s_done)

                # batch에 쌓기
                self.sstates.append(state)
                self.sactions.append(short_act)
                self.std_targets.append(y_i)
                self.sadvantages.append(advantage)

                if len(self.sstates) == self.BATCH_SIZE_LS:
                    # print("*****************************HI Short_learn********************************")
                    if True: # 배우는 비율 맞추기 self.tradetrain * 10 > self.shorttrain
                        start = TIME.time()
                        # critic 학습
                        self.ShortAgent.critic_learn(tf.convert_to_tensor(self.sstates, dtype=tf.float32),
                                            tf.convert_to_tensor(self.std_targets,dtype=tf.float32))

                        # actor 학습
                        self.ShortAgent.actor_learn(tf.convert_to_tensor(self.sstates, dtype=tf.float32),
                                                tf.convert_to_tensor(self.sactions, dtype=tf.float32),
                                            tf.convert_to_tensor(self.sadvantages,dtype=tf.float32))
                        print("\nShortAgent_learn exec time=",TIME.time() - start)
                        self.shorttrain += 1
                    # batch 지우기
                    self.sstates, self.sactions, self.std_targets, self.sadvantages = [], [], [], []
                    
                # reward += s_reward
                state = s_next_state
                if s_done:
                    close_position = finish_info[0]
                    # 따로 reward를 계산해줘야 한다.
                    percent = - self.env.cal_percent(position, close_position) - self.env.TRANS_FEE * 2
                    self.env.budget *= (1+percent)
                    reward = cal_reward(percent)
                    break # short 거래가 끝났으므로 빠져나간다.

            obs = state
            info = [position, close_position, self.env.curr]
            return obs, reward, done, info
        
        # 관망(2)이면
        # next_obs 그냥 다음 행 주기.
        else:
            _, obs = self.env.get_next_row_obs()
            if obs is None: # 판단할 수 없음. 버려야 함.
                reward = None
                done = self.env.ticker_is_done()
            else:
                self.decay = self.decay * STAY_DECAY
                reward = STAY_REWARD * self.decay
                done = self.env.ticker_is_done()

            return obs, reward, done, info
        
    def test(self, max_episode_num):

        #test_timestamp = TEST_TIMESTAMP
        test_timestamp = XLM_TIMESTAMP

        print("Test Start")

        for ep in range(int(max_episode_num)):

            info_list = []

            # 에피소드 리셋
            time, episode_reward, done = 0, 0, False

            # Env 첫 state 가져오기
            _, state = self.env.reset(test=True)

            lowest_budget = self.env.budget
            highest_budget = self.env.budget

            # 테스트 할 곳까지 curr 옮기기
            self.env.set_curr_to_timestamp(test_timestamp)

            while not done:

                act = self.TradeAgent.get_action(state)
                raw_act = act
                act = np.argmax(act)
                # 만약 테스트할 때 일정 확률 이상일 때만 실행시키면??
                if raw_act[act] < 0.9:
                    # print("pass")
                    _, state = self.env.get_next_row_obs()
                    if self.env.ticker_is_done():
                        break
                    continue

                # print(act)
                next_state, reward, done, info = self.test_action_step(act)
                if lowest_budget > self.env.budget:
                    lowest_budget = self.env.budget
                if highest_budget < self.env.budget:
                    highest_budget = self.env.budget
                
                if act < 2:
                    info_list.append(info)
                else:
                    # budget_list.append(info[3])
                    pass

                if done == True:
                    break

                if reward is None:
                    _, state = self.env.get_next_row_obs()
                    continue
                if act < 2:
                    print("time=",time, "raw-act=",raw_act ," act=",act," reward=",reward*100," open=",info[0]," close=",info[1], " curr=",info[2]," budget=",self.env.budget)

                state = next_state
                if done == True:
                    break

            # Test 결과 저장하기
            with open('./save_weights/test_reward.txt', 'a') as f:
                f.write(f"ep{ep} ticker = {tickers[self.env.curr_ticker]} budget = {self.env.budget} low_b = {lowest_budget} higi_b = {highest_budget}\n")

            # 거래 info 저장하기
            df = pd.DataFrame(info_list, columns=['open', 'close', 'end', 'start', 'budget', 'percent','position'])
            print(df)
            df.to_csv(f"./save_weights/{tickers[self.env.curr_ticker]}_test_result.csv")

        print("Test Finished")

    def test_action_step(self, action):

        done = False
        info = ["error", "sorry", self.env.curr]
        start_curr = self.env.curr

        # action이 매수(0, 1)이면
        if action == 0 or action == 1:
            ohlcv, state = self.env.get_next_row_obs()
            # state가 존재하지 않을 때
            if state is None:
                reward = None

                # ticker가 끝났는가?
                if self.env.ticker_is_done():
                    done = True # ticker끝
                else:
                    done = False

                info = ["obs is none"]
                return None, None, done, info # None, None, True, _
            
            reward = 0
            position = ohlcv['open']
            close_position = None

            while(1): # 포지션이 종료될 때 까지
                if action == 0:
                    act = self.LongAgent.get_action(state)
                else:
                    act = self.ShortAgent.get_action(state)
                act = np.argmax(act)
                if action == 0:
                    is_short = False
                else:
                    is_short = True
                next_state, reward, t_done, finish_info = self.env.long_or_short_step(act, position, is_short)

                if next_state is None:
                    reward = None
                    # ticker가 끝났는가?
                    if self.env.ticker_is_done():
                        done = True # ticker끝
                    else:
                        done = False
                    info = ["obs is none"]
                    return None, None, done, info # None, None, True, _
                
                state = next_state

                if t_done:
                    close_position = finish_info[0]
                    # 따로 reward를 계산해줘야 한다.
                    percent = self.env.cal_percent(position, close_position)
                    if action==1:
                        percent = -percent
                    percent = percent - self.env.TRANS_FEE * 2
                    
                    self.env.budget *= (1+percent)
                    break # 거래가 끝났으므로 빠져나간다.
                
            obs = state
            info = [position, close_position, self.env.curr]
            ### 데이터 추가
            # 모델의 거래별 홀딩 시간
            end_curr = self.env.curr
            # hold_time = end_curr - start_curr
            info.append(start_curr)
            # 모델의 budget
            info.append(self.env.budget)
            # 모델의 거래 손익 %
            info.append(percent * 100)
            # 모델의 포지션 ( Long, Short )
            info.append(action)
            
            return obs, percent, done, info
        # 관망(2)이면
        # 그냥 다음행 주기
        else:
            _, obs = self.env.get_next_row_obs()
            done = self.env.ticker_is_done()
            info = ["STAY", "HOLD", self.env.curr]
            # 모델의 budget
            info.append(self.env.budget)
            # 모델의 거래 손익 %
            return obs, 0, done, info
        



def test():
    agent = Train(time_steps=0, agent_type=ALL_TYPE)

    agent.load_weights("save_weights/a2c_17")
    agent.test(len(tickers))
    
def main():
    max_episode_num = 25
    agent = Train(time_steps=0, agent_type=ALL_TYPE)
    agent.load_weights("save_weights/a2c_17")

    agent.train(max_episode_num)

if __name__=='__main__':
    test()
        
            
