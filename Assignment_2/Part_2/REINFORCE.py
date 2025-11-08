from math import log
import gymnasium as gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch, random
from torch import nn
import copy
import tqdm
import matplotlib.pyplot as plt
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="cartpole") 
args = parser.parse_args()

SEED = 1
t = utils.torch.TorchHelper()
DEVICE = t.device

# 環境參數
if args.mode == "cartpole":
    OBS_N = 4 # 狀態空間
    ACT_N = 2 # 動作空間
    ENV_NAME = "CartPole-v0"
    GAMMA = 1.0 # 獎勵的折扣因子
    LEARNING_RATE = 5e-4 # 學習率(alpha)
elif "mountain_car" in args.mode:
    OBS_N = 2 # 狀態空間
    ACT_N = 3 # 動作空間
    ENV_NAME = "MountainCar-v0"
    GAMMA = 0.9 # 獎勵的折扣因子
    LEARNING_RATE = 1e-3 # 學習率(alpha)

EPOCHS = 800           # 總學習迭代次數
EPISODES_PER_EPOCH = 1  # 每個epoch的episode數
TEST_EPISODES = 10      # 測試的episode數
HIDDEN = 32             # 隱藏層大小
POLICY_TRAIN_ITERS = 1  # 每個epoch中策略改進的迭代次數

# 建置環境
utils.seed.seed(SEED)
env = gym.make(ENV_NAME)
env.reset(seed = SEED)

# 網路模型
pi = torch.nn.Sequential(
    torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, ACT_N)
).to(DEVICE)

# 優化器
OPT = torch.optim.Adam(pi.parameters(), lr = LEARNING_RATE)

# 策略(採用softmax)
def policy(env, obs):
    probs = torch.nn.Softmax(dim=-1)(pi(t.f(obs)))
    return np.random.choice(ACT_N, p = probs.cpu().detach().numpy())

# 訓練函式
def train(S, A, returns):
    for i in range(POLICY_TRAIN_ITERS):
        OPT.zero_grad()
        
        # 收集軌跡中 pi(a | s) 的機率
        log_probs = torch.nn.LogSoftmax(dim=-1)(pi(S)).gather(1, A.view(-1, 1)).view(-1)
        
        n = torch.arange(S.size(0)).to(DEVICE)
        objective = -((GAMMA**n) * returns * log_probs).sum()
        objective.backward()
        OPT.step()

# 進行訓練
Rs = [] 
last25Rs = []
print("Training:")
pbar = tqdm.trange(EPOCHS)
for epi in pbar:

    all_S, all_A = [], []
    all_returns = []

    for epj in range(EPISODES_PER_EPOCH):
        
        S, A, R = utils.envs.play_episode(env, policy)

        if args.mode == "mountain_car_mod":
            R = [s[0] for s in S[:-1]]


        all_S += S[:-1]
        all_A += A
        
        discounted_rewards = copy.deepcopy(R)
        for i in range(len(R)-1)[::-1]:
            discounted_rewards[i] += GAMMA * discounted_rewards[i+1]
        discounted_rewards = t.f(discounted_rewards)
        all_returns += [discounted_rewards]

    Rs += [sum(R)]
    S, A = t.f(np.array(all_S)), t.l(np.array(all_A))
    returns = torch.cat(all_returns, dim=0).flatten()

    train(S, A, returns)

    last25Rs += [sum(Rs[-25:])/len(Rs[-25:])]
    pbar.set_description("R25(%g, mean over 10 episodes)" % (last25Rs[-1]))
  
pbar.close()
print("Training finished!")

# 繪圖
N = len(last25Rs)
plt.plot(range(N), last25Rs, 'b')
plt.xlabel('Episode')
plt.ylabel('Reward (averaged over last 25 episodes)')
plt.title("REINFORCE, mode: " + args.mode)
plt.show()

# 進行測試
print("Testing:")
testRs = []
for epi in range(TEST_EPISODES):
    S, A, R = utils.envs.play_episode(env, policy, render = False)

    if "mountain_car" in args.mode:
        R = [s[0] for s in S[:-1]]

    testRs += [sum(R)]
    print("Episode%02d: R = %g" % (epi+1, sum(R)))

if "mountain_car" in args.mode:
    print("Height achieved: %.2f ± %.2f" % (np.mean(testRs), np.std(testRs)))
else:
    print("Eval score: %.2f ± %.2f" % (np.mean(testRs), np.std(testRs)))

env.close()