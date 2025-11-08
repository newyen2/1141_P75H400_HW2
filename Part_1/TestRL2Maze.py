import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt

'''
  ---------------------
  |  0 |  1 |  2 |  3 |
  ---------------------
  |  4 |  5 |  6 |  7 |
  ---------------------
  |  8 |  9 | 10 | 11 |
  ---------------------
  | 12 | 13 | 14 | 15 |
  ---------------------

  終點狀態: 14 
  壞狀態: 9
  結束狀態: 16

  到達終點狀態後下回合會進入結束狀態 '''

T = np.zeros([4,17,17])
a = 0.8  # 預期移動
b = 0.1  # 偏移移動

# 定義轉移機率
def init_transition():
    T[0, 0, 0] = a + b
    T[0, 0, 1] = b

    T[0, 1, 0] = b
    T[0, 1, 1] = a
    T[0, 1, 2] = b

    T[0, 2, 1] = b
    T[0, 2, 2] = a
    T[0, 2, 3] = b

    T[0, 3, 2] = b
    T[0, 3, 3] = a + b

    T[0, 4, 4] = b
    T[0, 4, 0] = a
    T[0, 4, 5] = b

    T[0, 5, 4] = b
    T[0, 5, 1] = a
    T[0, 5, 6] = b

    T[0, 6, 5] = b
    T[0, 6, 2] = a
    T[0, 6, 7] = b

    T[0, 7, 6] = b
    T[0, 7, 3] = a
    T[0, 7, 7] = b

    T[0, 8, 8] = b
    T[0, 8, 4] = a
    T[0, 8, 9] = b

    T[0, 9, 8] = b
    T[0, 9, 5] = a
    T[0, 9, 10] = b

    T[0, 10, 9] = b
    T[0, 10, 6] = a
    T[0, 10, 11] = b

    T[0, 11, 10] = b
    T[0, 11, 7] = a
    T[0, 11, 11] = b

    T[0, 12, 12] = b
    T[0, 12, 8] = a
    T[0, 12, 13] = b

    T[0, 13, 12] = b
    T[0, 13, 9] = a
    T[0, 13, 14] = b

    T[0, 14, 16] = 1

    T[0, 15, 11] = a
    T[0, 15, 14] = b
    T[0, 15, 15] = b

    T[0, 16, 16] = 1

    # down (a = 1)

    T[1, 0, 0] = b
    T[1, 0, 4] = a
    T[1, 0, 1] = b

    T[1, 1, 0] = b
    T[1, 1, 5] = a
    T[1, 1, 2] = b

    T[1, 2, 1] = b
    T[1, 2, 6] = a
    T[1, 2, 3] = b

    T[1, 3, 2] = b
    T[1, 3, 7] = a
    T[1, 3, 3] = b

    T[1, 4, 4] = b
    T[1, 4, 8] = a
    T[1, 4, 5] = b

    T[1, 5, 4] = b
    T[1, 5, 9] = a
    T[1, 5, 6] = b

    T[1, 6, 5] = b
    T[1, 6, 10] = a
    T[1, 6, 7] = b

    T[1, 7, 6] = b
    T[1, 7, 11] = a
    T[1, 7, 7] = b

    T[1, 8, 8] = b
    T[1, 8, 12] = a
    T[1, 8, 9] = b

    T[1, 9, 8] = b
    T[1, 9, 13] = a
    T[1, 9, 10] = b

    T[1, 10, 9] = b
    T[1, 10, 14] = a
    T[1, 10, 11] = b

    T[1, 11, 10] = b
    T[1, 11, 15] = a
    T[1, 11, 11] = b

    T[1, 12, 12] = a + b
    T[1, 12, 13] = b

    T[1, 13, 12] = b
    T[1, 13, 13] = a
    T[1, 13, 14] = b

    T[1, 14, 16] = 1

    T[1, 15, 14] = b
    T[1, 15, 15] = a + b

    T[1, 16, 16] = 1

    # left (a = 2)

    T[2, 0, 0] = a + b
    T[2, 0, 4] = b

    T[2, 1, 1] = b
    T[2, 1, 0] = a
    T[2, 1, 5] = b

    T[2, 2, 2] = b
    T[2, 2, 1] = a
    T[2, 2, 6] = b

    T[2, 3, 3] = b
    T[2, 3, 2] = a
    T[2, 3, 7] = b

    T[2, 4, 0] = b
    T[2, 4, 4] = a
    T[2, 4, 8] = b

    T[2, 5, 1] = b
    T[2, 5, 4] = a
    T[2, 5, 9] = b

    T[2, 6, 2] = b
    T[2, 6, 5] = a
    T[2, 6, 10] = b

    T[2, 7, 3] = b
    T[2, 7, 6] = a
    T[2, 7, 11] = b

    T[2, 8, 4] = b
    T[2, 8, 8] = a
    T[2, 8, 12] = b

    T[2, 9, 5] = b
    T[2, 9, 8] = a
    T[2, 9, 13] = b

    T[2, 10, 6] = b
    T[2, 10, 9] = a
    T[2, 10, 14] = b

    T[2, 11, 7] = b
    T[2, 11, 10] = a
    T[2, 11, 15] = b

    T[2, 12, 8] = b
    T[2, 12, 12] = a + b

    T[2, 13, 9] = b
    T[2, 13, 12] = a
    T[2, 13, 13] = b

    T[2, 14, 16] = 1

    T[2, 15, 11] = b
    T[2, 15, 14] = a
    T[2, 15, 15] = b

    T[2, 16, 16] = 1

    # right (a = 3)

    T[3, 0, 0] = b
    T[3, 0, 1] = a
    T[3, 0, 4] = b

    T[3, 1, 1] = b
    T[3, 1, 2] = a
    T[3, 1, 5] = b

    T[3, 2, 2] = b
    T[3, 2, 3] = a
    T[3, 2, 6] = b

    T[3, 3, 3] = a + b
    T[3, 3, 7] = b

    T[3, 4, 0] = b
    T[3, 4, 5] = a
    T[3, 4, 8] = b

    T[3, 5, 1] = b
    T[3, 5, 6] = a
    T[3, 5, 9] = b

    T[3, 6, 2] = b
    T[3, 6, 7] = a
    T[3, 6, 10] = b

    T[3, 7, 3] = b
    T[3, 7, 7] = a
    T[3, 7, 11] = b

    T[3, 8, 4] = b
    T[3, 8, 9] = a
    T[3, 8, 12] = b

    T[3, 9, 5] = b
    T[3, 9, 10] = a
    T[3, 9, 13] = b

    T[3, 10, 6] = b
    T[3, 10, 11] = a
    T[3, 10, 14] = b

    T[3, 11, 7] = b
    T[3, 11, 11] = a
    T[3, 11, 15] = b

    T[3, 12, 8] = b
    T[3, 12, 13] = a
    T[3, 12, 12] = b

    T[3, 13, 9] = b
    T[3, 13, 14] = a
    T[3, 13, 13] = b

    T[3, 14, 16] = 1

    T[3, 15, 11] = b
    T[3, 15, 15] = a + b

    T[3, 16, 16] = 1

init_transition()

# 定義獎勵函數，預設為-1
R = -1 * np.ones([4,17])

# 抵達終點狀態: 100, 抵達壞狀態: -70, 抵達結束狀態: 0
R[:,14] = 100  # goal state
R[:,9] = -70   # bad state
R[:,16] = 0    # end state

# 折扣係數
discount = 0.95
        
# MDP
mdp = MDP.MDP(T,R,discount)

# RL problem
rlProblem = RL2.RL2(mdp,np.random.normal)

# model-based RL
[V, policy, avg1] = rlProblem.modelBasedRL(s0=0,defaultT=np.ones([mdp.nActions,mdp.nStates,mdp.nStates])/mdp.nStates,initialR=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.05)

# Q-learning
[Q, policy, avg2] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.05)

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

avg1 = running_mean(avg1, 10)
avg2 = running_mean(avg2, 10)

episodes = np.arange(11, len(avg2) + 11)
plt.figure(figsize=(9,5))
plt.plot(episodes, avg1, label='Model-based RL')
plt.plot(episodes, avg2, label='Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Cumulative discounted rewards')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
