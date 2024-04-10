from hybrid_agent import Hypridagent
import os
import tensorflow as tf 
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():

    max_episode_num = 20
    agent = Hypridagent()

    agent.train(max_episode_num)

if __name__=='__main__':
    print(device_lib.list_local_devices() )
    main()


