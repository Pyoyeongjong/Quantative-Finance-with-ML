import gym
from hybrid_agent import Hypridagent

def main():

    max_episode_num = 200
    agent = Hypridagent()

    agent.train(max_episode_num)

if __name__=='__main__':
    main()


