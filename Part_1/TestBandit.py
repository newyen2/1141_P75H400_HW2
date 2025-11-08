import numpy as np
import MDP
import RL2
from tqdm import tqdm
import matplotlib.pyplot as plt



def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

# Test epsilon greedy strategy
avg_rewards1 = np.zeros(200)
for i in tqdm(range(1000)):
    empiricalMeans, rewards = banditProblem.epsilonGreedyBandit(nIterations=200)
    avg_rewards1 += rewards
avg_rewards1 /= 1000

# Test Thompson sampling strategy
avg_rewards2 = np.zeros(200)
for i in tqdm(range(1000)):
    empiricalMeans, rewards = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
    avg_rewards2 += rewards
avg_rewards2 /= 1000

# Test UCB strategy
avg_rewards3 = np.zeros(200)
for i in tqdm(range(1000)):
    empiricalMeans, rewards = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
    avg_rewards3 += rewards
avg_rewards3 /= 1000

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

avg_rewards1 = running_mean(avg_rewards1, 10)
avg_rewards2 = running_mean(avg_rewards2, 10)
avg_rewards3 = running_mean(avg_rewards3, 10)

episodes = np.arange(11, len(avg_rewards1) + 11)
plt.figure(figsize=(9,5))
plt.plot(episodes, avg_rewards1, label='Epsilon Greedy')
plt.plot(episodes, avg_rewards2, label='Thompson Sampling')
plt.plot(episodes, avg_rewards3, label='UCB')
plt.xlabel('Episode')
plt.ylabel('Cumulative discounted rewards')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
