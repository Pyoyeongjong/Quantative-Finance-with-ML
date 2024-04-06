import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential
from replaybuffer import ReplayBuffer
from bitcoinenv import BitcoinTradingEnv
from gym.spaces import Discrete, Space

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size # observation size
        self.action_size = action_size # 행동 사이즈
        self.action_low = action_low # 각 행동의 범위
        self.action_high = action_high
        self.model = self.build_model()

    def build_model(self):
        states = keras.Input(shape=(self.state_size,))
        out = layers.Dense(400, activation="relu")(states)
        out = layers.Dense(300, activation="relu")(out)
        raw_actions = layers.Dense(self.action_size, activation="tanh")(out)
        # raw action을 action bound에 맞춰준 것.
        actions = layers.Lambda(lambda x: (x *(self.action_high - self.action_low) / 2) + (self.action_high + self.action_low) / 2)(raw_actions)
        model = keras.Model(inputs=states, outputs=actions)
        return model


class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        states = keras.Input(shape=(self.state_size,))
        actions = keras.Input(shape=(self.action_size,))
        out = layers.Concatenate()([states, actions])
        out = layers.Dense(400, activation="relu")(out)
        out = layers.Dense(300, activation="relu")(out)
        q_values = layers.Dense(1, activation=None)(out)
        model = tf.keras.Model(inputs=[states, actions], outputs=q_values)
        return model


def ou_noise(x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
    return x + rho*(mu-x)*dt + sigma*np.sqrt(dt)*np.sqrt(dt)*np.random.normal(size=dim)


low_arr = np.asarray([0, 0.08 * 0.01, 0])
high_arr = np.asarray([0.9, 1000, 1])

class DDPGagent:

    def __init__(self, state_dim, action_dim):
        # hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        self.state_dim = state_dim# env.observation + ddqn action
        self.action_dim = action_dim# (손절가, 익절가, 사이즈)
        self.action_low = low_arr
        self.action_high = high_arr

        self.actor = Actor(self.state_dim, self.action_dim, self.action_low, self.action_high)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_low, self.action_high)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)

        self.actor_opt = Adam(learning_rate=self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(learning_rate=self.CRITIC_LEARNING_RATE)

        # show
        self.actor.model.summary()
        self.critic.model.summary()


    def update_target_network(self, TAU):
        theta = self.actor.model.get_weights()
        target_theta = self.target_actor.model.get_weights()

        for i in range(len(theta)):
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        self.target_actor.model.set_weights(target_theta)

        phi = self.critic.model.get_weights()
        target_phi = self.target_critic.model.get_weights()

        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.model.set_weights(target_phi)

    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            q = self.critic.model([states, actions], training=True)  # q = critic으로 가져온 결과값
            loss = tf.reduce_mean(tf.square(q - td_targets))  # q - td_targets 을 통해 loss 계산

        grads = tape.gradient(loss, self.critic.model.trainable_variables)  # loss로 gradient 계산
        self.critic_opt.apply_gradients(zip(grads, self.critic.model.trainable_variables))  # critic 조정

    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor.model(states, training=True)  # states에 대한 action을 뽑아서
            critic_q = self.critic.model([states, actions])  # critic함수에 넣고 가치평가 하고
            loss = -tf.reduce_mean(critic_q)  # critic_q에 대한 loss계산

        grads = tape.gradient(loss, self.actor.model.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.model.trainable_variables))

    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k

    # state에 따라 행동하는 파트
    def act(self, state, act):
        # 이거 해야함
        real_state = np.append(state, act)
        print(real_state)
        real_state = tf.expand_dims(real_state, axis=0)
        action = self.actor.model(tf.convert_to_tensor([real_state], dtype=tf.float32))
        print(action)
        return action

class DDQNagent:

    def __init__(self, state_size, action_size):

        # hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.TAU = 0.01

        # model parameters
        self.state_size = state_size# observation 사이즈
        self.action_size = action_size# 3개의 이산적인 사이즈
        self.model = self._build_model()
        self.target_model = self._build_model()

        # show
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(layers.Dense(32, input_dim=self.state_size, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="linear"))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self, TAU):
        phi = self.model.get_weights()
        target_phi = self.target_model.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_model.set_weights(target_phi)
        # 이건 targetNetwork 존재의믜가 없는 코드. 손좀 봐줄 필요있다.

    def model_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_size)
            q = self.model(states, training=True)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(q_values - td_targets))

    # 미래 보상을 반영하기 위해서 쓰는 함수다..
    def td_target(self, rewards, target_qs, max_a, dones):
        one_hot_max_a = tf.one_hot(max_a, self.action_size) # max_a 자리만 1이고 나머지 0으로 만드는 벡터를 생성
        max_q = tf.reduce_sum(one_hot_max_a * target_qs, axis=1, keepdims=True)
        y_k = np.zeros(max_q.shape)
        for i in range(max_q.shape[0]): # 배치 내 샘플 개수
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.gamma * max_q[i]
            return y_k

    # state에 따라 행동하는 파트
    def act(self, state, env):
        env = env
        # 이거 해야함
        if np.random.random() <= self.epsilon:
            return env.action_space.sample()
        else:
            qs = self.model(tf.convert_to_tensor(state, dtype=tf.float32))

            return np.argmax(qs.numpy())


class Hypridagent:

    def __init__(self):
        self.env = BitcoinTradingEnv()
        self.DDQN = DDQNagent(self.env.observation_space.shape[0], 3)
        self.DDPG = DDPGagent(self.env.observation_space.shape[0] + 1, 3)
        self.BUFFER_SIZE = 20000
        self.BATCH_SIZE = 32
        # Replay Buffer
        self.ddqnbuffer = ReplayBuffer(self.BUFFER_SIZE)
        self.ddpgbuffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []

    def train(self, max_episode_num):

        self.DDQN.update_target_network(1.0)
        self.DDPG.update_target_network(1.0)

        for ep in range(int(max_episode_num)):

            # DDPG ou-noise 리셋
            pre_noise = np.zeros(self.DDPG.action_dim)

            # 에피소드 리셋
            time, episode_reward, done = 0, 0, False

            # Env 첫 state 가져우기
            state = self.env.reset()

            while not done:
                # state에 따른 행동 뽑아오기
                ddqn_act = self.DDQN.act(state, self.env)
                print(ddqn_act[0])
                ddpg_act = self.DDPG.act(state, ddqn_act[0])
                print(ddpg_act)
                noise = ou_noise(pre_noise, dim=self.DDPG.action_dim)
                ddpg_act = np.clip(ddpg_act + noise, self.DDPG.action_low, self.DDPG.action_high)

                action = [ddqn_act, ddpg_act]

                # act를 env에 넣기
                next_state, reward, done, _ = self.env.step(action) # env의 action 입력값은 [] 리스트

                # reward 손좀 보기
                train_reward = reward

                # 버퍼에 기록하기 - DDQN
                self.ddqnbuffer.add_buffer(state, action[0], train_reward, next_state, done) # 여기서 next_state가 학습에 영향을 줄까??

                # 버퍼에 기록하기 - DDPG
                ddpg_next_act = self.DDQN.act(next_state)
                # state를 이렇게 저장해도 되는가?
                self.ddpgbuffer.add_buffer([state, action[0]], action[1], train_reward, [next_state, ddpg_next_act], done)

                if self.ddqnbuffer.buffer_count() > 100 and self.ddpgbuffer.buffer_count() > 100:

                    # DDQN epsilon decaying
                    if self.DDQN.epsilon < self.DDQN.epsilon_min:
                        self.DDQN.epsilon *= self.DDQN.epsilon_decay

                    # replay buffer로부터 sample 뽑기 - DDQN
                    states1, actions1, rewards1, next_states1, dones1 = self.ddqnbuffer.sample_batch(self.BATCH_SIZE)

                    ### DDQN 학습
                    # max_a 계산
                    curr_net_qs = self.DDQN.model(tf.convert_to_tensor(next_states1,dtype=tf.float32))
                    max_a = np.argmax(curr_net_qs.numpy(),axis=1)
                    # target-Q value 예측
                    ddqn_target_qs = self.DDQN.target_model(tf.convert_to_tensor(next_states1, dtype=tf.float32))
                    # TD target 계산
                    y_i = self.DDQN.td_target(rewards1, ddqn_target_qs.numpy(), max_a, dones1)
                    # machine learning
                    self.DDQN.model_learn(tf.convert_to_tensor(next_states1, dtype=tf.float32),
                                          actions1,
                                          tf.convert_to_tensor(y_i, dtype=tf.float32))
                    # target network 업데이트
                    self.DDQN.update_target_network(self.DDQN.TAU)

                    # replay buffer로부터 sample 뽑기 - DDPG
                    states2, actions2, rewards2, next_states2, dones2 = self.ddpgbuffer.sample_batch(self.BATCH_SIZE)

                    ### DDPG 학습
                    # action[0]도 state에 포함시켜야 한다.
                    #ddpg_state = [state, action[0]]
                    # target-Q value 예측 -> next_state에서는 action[0]이 없는데 어떻게 하지...??
                    # 버퍼를 따로 뽑아야할 듯

                    ddpg_target_qs = self.DDPG.target_critic.model([tf.convert_to_tensor(next_states1, dtype=tf.float32),
                                                              self.DDPG.target_actor.model(tf.convert_to_tensor(next_states2, dtype=tf.float32))])
                    # TD target 계산
                    y_i = self.DDPG.td_target(rewards2, ddpg_target_qs.numpy(), dones2)
                    # machine learning
                    self.DDPG.critic_learn(tf.convert_to_tensor(states2, dtype=tf.float32),
                                           tf.convert_to_tensor(actions2, dtype=tf.float32),
                                           tf.convert_to_tensor(y_i,dtype=tf.float32))
                    self.DDPG.actor_learn(tf.convert_to_tensor(states2, dtype=tf.float32))
                    # target network 업데이트
                    self.DDPG.update_target_network(self.DDPG.TAU)

                # 현재 상태 업데이트하기
                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            # display each episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            ## 각 weight 저장하기
            # 저장 코드

        # epi_reward 저장
        # 저장코드














