import tensorflow as tf
import numpy as np
import keras
from keras import layers, initializers
from keras.optimizers import Adam
from keras.models import Sequential
from replaybuffer import ReplayBuffer
from bitcoinA2Cenv import BitcoinTradingEnv
from gym.spaces import Discrete, Space
import matplotlib.pyplot as plt

class Actor:
    def __init__(self, state_size, action_dim):
        self.state_size = state_size # observation size
        self.action_dim = action_dim # 행동 사이즈
        self.model = self.build_model()

    def build_model(self):
        states = keras.Input(shape=(self.state_size,))
        out = layers.Dense(256, activation="relu",kernel_initializer=initializers.HeNormal())(states)
        out = layers.Dense(128, activation="relu",kernel_initializer=initializers.HeNormal())(out)
        actions = layers.Dense(self.action_dim, activation='softmax')(out)
        
        model = keras.Model(inputs=states, outputs=actions)
        return model


class Critic:
    def __init__(self, state_size):
        self.state_size = state_size

        self.model = self.build_model()

    def build_model(self):
        states = keras.Input(shape=(self.state_size,))
        out = layers.Dense(256, activation="relu",kernel_initializer=initializers.HeNormal())(states)
        out = layers.Dense(128, activation="relu",kernel_initializer=initializers.HeNormal())(out)
        q_values = layers.Dense(1, activation='linear')(out)
        model = keras.Model(inputs=states, outputs=q_values)
        return model

class A2Cagent:

    def __init__(self, state_dim, action_dim):
        # hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor_opt = Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        self.save_epi_reward = []

    # action 뱉기
    def get_action(self, state):
        state = np.array([state])
        # print("action_state=", state)
        action_probs = self.actor.model(state)
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
            loss = -tf.reduce_mean(action_probs + 1e-10) * advantages # critic_q에 대한 loss계산

        grads = tape.gradient(loss, self.actor.model.trainable_variables)
        # Gradient Cliping
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.actor_opt.apply_gradients(zip(grads, self.actor.model.trainable_variables))



class Train:

    def __init__(self):
        self.env = BitcoinTradingEnv()
        self.TradeAgent = A2Cagent(self.env.observation_space.shape[0], 3)
        self.LongAgent = A2Cagent(self.env.observation_space.shape[0], 2)
        self.ShortAgent = A2Cagent(self.env.observation_space.shape[0], 2)
        self.BATCH_SIZE = 32
        
        # long ,short agent 용 batch들
        self.lstates, self.lactions, self.ltd_targets, self.ladvantages = [], [], [], []
        self.sstates, self.sactions, self.std_targets, self.sadvantages = [], [], [], []

        self.save_epi_reward = []
        print("Trainer OK")

    

    def train(self, max_episode_num):

        print("Train start")

        # ep 한 번 = 한 ticker를 한바퀴 돌았을 때
        for ep in range(int(max_episode_num)):

            # initialize batch
            states, actions, td_targets, advantages = [], [], [], []

            # 에피소드 리셋
            time, episode_reward, done = 0, 0, False

            # Env 첫 state 가져오기
            _, state = self.env.reset()

            while not done:
                
                ta_act = self.TradeAgent.get_action(state)
                act = np.random.choice(len(ta_act), p=ta_act)
                next_state, reward, done, info = self.action_step(act) 
                
                # budget 계산을 여기서 하자
                if reward >= 0:
                    self.env.budget = self.env.budget*(1+reward/100)
                else:
                    self.env.budget = self.env.budget*(1+reward/100)
                # 여기서 done은 obs가 None일 떄(마지막일 떄) 만 True이다.

                
                # debug용
                print("time=",time," act=",act," reward=",reward," open=",info[0]," close=",info[1], " curr=",info[2]," budget=",self.env.budget)

                if done == True: # 끝났다는 뜻
                    break

                # obs None (ticker가 끝남)
                if next_state.any() == None:
                    done = True # ticker끝
                    break

                # TradeAgent학습
                v_value = self.TradeAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                next_v_value = self.TradeAgent.critic.model(tf.convert_to_tensor(np.array([next_state]), dtype=tf.float32))
                train_reward = (reward+8)/8
                # 사실 done이 True면 가치평가를 할 수가 없긴 함
                advantage, y_i = self.TradeAgent.td_target(train_reward, v_value, next_v_value, done)

                # batch에 쌓기
                states.append(state)
                actions.append(ta_act)
                td_targets.append(y_i)
                advantages.append(advantage)

                if len(states) == self.BATCH_SIZE:
                    print("*****************************HI Action_learn********************************")
                    # critic 학습
                    self.TradeAgent.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                           tf.convert_to_tensor(td_targets,dtype=tf.float32))

                    # actor 학습
                    self.TradeAgent.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                               tf.convert_to_tensor(actions, dtype=tf.float32),
                                           tf.convert_to_tensor(td_targets,dtype=tf.float32))

                    # batch 지우기
                    states, actions, td_targets, advantages = [], [], [], []
                    

                episode_reward += reward
                state = next_state
                time += 1

                if done:
                    break # long 거래가 끝났으므로 빠져나간다.

            # while 밖
            # display each episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            ## 각 weight 저장하기
            # 저장 코드

            self.TradeAgent.actor.model.save_weights("./save_weights/Trade_actor.weights.h5")
            self.TradeAgent.critic.model.save_weights("./save_weights/Trade_critic.weights.h5")
            self.LongAgent.actor.model.save_weights("./save_weights/Long_actor.weights.h5")
            self.LongAgent.critic.model.save_weights("./save_weights/Long_critic.weights.h5")
            self.ShortAgent.actor.model.save_weights("./save_weights/Short_actor.weights.h5")
            self.ShortAgent.critic.model.save_weights("./save_weights/Short_critic.weights.h5")
            print("[Save Weight]: ep",ep," save Completed")

            with open('./save_weights/train_reward.txt', 'a') as f:
                f.write(f"ep{ep} reward = {episode_reward}\n")
        
        # for 밖

    # obs 가 none이 불리면, agent가 reset을 진행.
    def action_step(self, action):

        done = False
        info = ["error", "sorry", self.env.curr]

        # action 이 매수(0)이면
        # long_step을 통해 진행
        if action == 0:

            ohlcv, state = self.env.get_next_row_obs()
            reward = 0
            position = ohlcv['close'] # 임시
            close_position = None

            while(1): # long action이 종료될 때 까지

                long_act = self.LongAgent.get_action(state)
                act = np.random.choice(len(long_act), p=long_act)
                # obs가 None이거나, 거래거 끝났을 때 done = True
                l_next_state, l_reward, l_done, finish_info = self.env.long_step(act, position)

                # obs None (ticker가 끝남)
                if l_next_state.any() == None:
                    reward = l_reward # None
                    done = True # ticker끝
                    info = "obs is none"
                    return l_next_state, reward, done, info # None, None, True, _

                # LongAgent학습 (Main Agent와 빈도를 맞춰줘야 할 것 같다.)
                v_value = self.LongAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                next_v_value = self.LongAgent.critic.model(tf.convert_to_tensor(np.array([l_next_state]), dtype=tf.float32))
                train_reward = (l_reward + 8)/8
                advantage, y_i = self.LongAgent.td_target(train_reward, v_value, next_v_value, l_done)

                # batch에 쌓기
                self.lstates.append(state)
                self.lactions.append(long_act)
                self.ltd_targets.append(y_i)
                self.ladvantages.append(advantage)

                if len(self.lstates) == self.BATCH_SIZE:
                    print("*****************************HI Long_learn********************************")

                    # critic 학습
                    self.LongAgent.critic_learn(tf.convert_to_tensor(self.lstates, dtype=tf.float32),
                                           tf.convert_to_tensor(self.ltd_targets, dtype=tf.float32))

                    # actor 학습
                    self.LongAgent.actor_learn(tf.convert_to_tensor(self.lstates, dtype=tf.float32),
                                               tf.convert_to_tensor(self.lactions, dtype=tf.float32),
                                           tf.convert_to_tensor(self.ltd_targets,dtype=tf.float32))

                    # batch 지우기
                    self.lstates, self.lactions, self.ltd_targets, self.ladvantages = [], [], [], []
                    
                reward += l_reward
                state = l_next_state
                if l_done:
                    close_position = finish_info[0]
                    break # long 거래가 끝났으므로 빠져나간다.

            obs = state
            info = [position, close_position, self.env.curr]

            return obs, reward, done, info

        # action 이 매도(1)이면
        # short_step을 통해 진행
        elif action == 1:

            ohlcv, state = self.env.get_next_row_obs()
            reward = 0
            position = ohlcv['close'] # 임시
            close_position = None

            while(1): # shont action이 종료될 때 까지

                short_act = self.ShortAgent.get_action(state)
                act = np.random.choice(len(short_act), p=short_act)
                # obs가 None이거나, 거래거 끝났을 때 done = True
                s_next_state, s_reward, s_done, finish_info = self.env.short_step(act, position)
                

                # obs None (ticker가 끝남)
                if s_next_state.any() == None:
                    reward = s_reward # None
                    done = True # ticker끝
                    return s_next_state, reward, done, info # None, None, True, _

                # LongAgent학습 (Main Agent와 빈도를 맞춰줘야 할 것 같다.)
                v_value = self.ShortAgent.critic.model(tf.convert_to_tensor(np.array([state]), dtype=tf.float32))
                next_v_value = self.ShortAgent.critic.model(tf.convert_to_tensor(np.array([s_next_state]), dtype=tf.float32))
                train_reward = (s_reward+8)/8
                advantage, y_i = self.ShortAgent.td_target(train_reward, v_value, next_v_value, s_done)

                # batch에 쌓기
                self.sstates.append(state)
                self.sactions.append(short_act)
                self.std_targets.append(y_i)
                self.sadvantages.append(advantage)

                if len(self.sstates) == self.BATCH_SIZE:
                    print("*****************************HI Short_learn********************************")
                    print("open_price=",position,"curr_price=",finish_info[0])
                    # critic 학습
                    self.ShortAgent.critic_learn(tf.convert_to_tensor(self.sstates, dtype=tf.float32),
                                           tf.convert_to_tensor(self.std_targets,dtype=tf.float32))

                    # actor 학습
                    self.ShortAgent.actor_learn(tf.convert_to_tensor(self.sstates, dtype=tf.float32),
                                               tf.convert_to_tensor(self.sactions, dtype=tf.float32),
                                           tf.convert_to_tensor(self.std_targets,dtype=tf.float32))

                    # batch 지우기
                    self.sstates, self.sactions, self.std_targets, self.sadvantages = [], [], [], []
                    
                reward += s_reward
                state = s_next_state
                if s_done:
                    close_position = finish_info[0]
                    break # long 거래가 끝났으므로 빠져나간다.

            obs = state
            info = [position, close_position, self.env.curr]
            return obs, reward, done, info
        
        # 관망(2)이면
        # next_obs 그냥 다음 행 주기.
        else:
            _, obs = self.env.get_next_row_obs()
            if obs is None: # 판단할 수 없음.
                reward = None
                done = True
            else:
                reward = 0
                done = False

            return obs, reward, done, info
        
def main():

    max_episode_num = 200
    agent = Train()

    agent.train(max_episode_num)

if __name__=='__main__':
    main()
            
