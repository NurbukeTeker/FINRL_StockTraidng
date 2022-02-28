# import main_A2C
# import main_DDPG
# import main_SAC
# import main_TD3
# import main_PPO


import os

print("A2C")
os.system('python main_A2C.py')

print("DDPG")
os.system('python main_DDPG.py')

print("SAC")
os.system('python main_SAC.py')

print("TD3")
os.system('python main_TD3.py')

print("PPO")
os.system('python main_PPO.py')




# import tensorflow as tf
# tf.test.is_gpu_available()