import numpy as np
import MDP
from tqdm import tqdm
import random

class RL2:
    def __init__(self,mdp,sampleReward):
        '''
        Inputs:
        mdp -- Markov Decision Process (包含T, R, discount)
        sampleReward -- 取樣使用的函數 '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''
        給予當前狀態與選擇動作，取樣後得到獎勵與下個狀態
        Inputs:
        state -- 當前狀態
        action -- 選擇的動作

        Outputs: 
        reward -- 取樣所得的獎勵
        nextState -- MDP中轉移到的下個狀態
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]  

    def epsilonGreedyBandit(self,nIterations):
        '''
        Epsilon貪婪演算法，沒有折扣因子，epsilon = 1 / iteration

        Inputs:
        nIterations -- 迭代次數

        Outputs: 
        empiricalMeans -- 每個Arm的經驗平均獎勵: |S|
        rewards -- 獎勵的歷史紀錄: |nIteration|
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        counts = np.zeros(self.mdp.nActions)
        empiricalMeans = np.zeros(self.mdp.nActions)

        rewards = np.zeros(nIterations)

        state = 0

        for i in range(nIterations):
            epsilon = 1 / (i + 1) 

            p = np.random.random()
            if p < epsilon:
                action = np.random.randint(self.mdp.nActions)
            else:
                action = np.argmax(empiricalMeans)

            reward, state = self.sampleRewardAndNextState(state, action)
            rewards[i] = reward

            counts[action] += 1
            empiricalMeans[action] = empiricalMeans[action] + (reward - empiricalMeans[action]) / counts[action]

        return empiricalMeans, rewards

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''
        Thompson抽樣演算法

        Inputs:
        prior -- 每個Arm平均獎勵的Beta先驗分佈 : |A| * 2
        nIterations -- 迭代次數
        k -- 每個Arm抽樣的樣本數

        Outputs: 
        empiricalMeans -- 每個Arm的經驗平均獎勵: |S|
        rewards -- 獎勵的歷史紀錄: |nIteration|
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        counts = np.zeros(self.mdp.nActions)
        empiricalMeans = np.zeros(self.mdp.nActions)

        rewards = np.zeros(nIterations)

        state = 0

        for i in range(nIterations):
            sampleMeans = np.zeros(self.mdp.nActions)

            # 進行Thompson Sampling
            for action in range(self.mdp.nActions):
                sample = np.random.beta(a = prior[action, 0], b = prior[action, 1], size = k)
                sampleMeans[action] = np.mean(sample)

            action = np.argmax(sampleMeans)

            reward, state = self.sampleRewardAndNextState(state, action)
            rewards[i] = reward

            counts[action] += 1
            empiricalMeans[action] = empiricalMeans[action] + (reward - empiricalMeans[action]) / counts[action]
            
            # 後驗更新
            if reward == 1:
                prior[action, 0] += 1
            else:
                prior[action, 1] += 1

        return empiricalMeans, rewards

    def UCBbandit(self,nIterations):
        '''
        上置信界(UCB)演算法

        Inputs:
        nIterations -- 迭代次數

        Outputs: 
        empiricalMeans -- 每個Arm的經驗平均獎勵: |S|
        rewards -- 獎勵的歷史紀錄: |nIteration|
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        counts = np.zeros(self.mdp.nActions)
        empiricalMeans = np.zeros(self.mdp.nActions)

        rewards = np.zeros(nIterations)

        UCBs = np.full(self.mdp.nActions, float('inf')) # UCB初始設為Inf，以保證所有Arm都被選擇至少一次

        state = 0

        for i in range(nIterations):

            # 計算所有Arm的UCB
            for action in range(self.mdp.nActions):
                if counts[action] == 0:
                    continue
                UCBs[action] = empiricalMeans[action] + np.sqrt(2 * np.log(i + 1) / counts[action])
            
            action = np.argmax(UCBs) 

            reward, state = self.sampleRewardAndNextState(state, action)
            rewards[i] = reward

            counts[action] += 1
            empiricalMeans[action] = empiricalMeans[action] + (reward - empiricalMeans[action]) / counts[action]

        return empiricalMeans, rewards

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''
        Q-Learning函數

        Inputs:
        s0 -- 初始狀態
        initialQ -- 初始價值函數: Q[A, S]
        nEpisodes -- 執行回合數
        nSteps -- 每回合的執行步數
        epsilon -- 探索率
        temperature -- Boltzmann exploration探索的溫度參數

        Outputs: 
        Q -- 價值函數: Q[A, S]
        policy -- 最終策略
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        avg_rewards = np.zeros(nEpisodes)

        for trial in tqdm(range(100)):
            Q = initialQ.copy()
            policy = np.zeros(self.mdp.nStates, dtype=int)

            rewards = np.zeros(nEpisodes)

            for episode in range(nEpisodes):
                state = s0

                for step in range(nSteps):
                    if random.random() < epsilon:
                        action = np.random.choice(self.mdp.nActions)
                    elif temperature > 0:
                        Q_value = Q[:, state]
                        probs = (np.exp(Q_value/temperature) / np.sum(np.exp(Q_value/temperature)))
                        action = np.random.choice(len(Q_value), p=probs)
                    else:
                        Q_value = Q[:, state]
                        action = np.argmax(Q_value)

                    reward, s_next = self.sampleRewardAndNextState(state, action)

                    Q[action, state] += 0.1 * (reward + self.mdp.discount * np.max(Q[:, s_next]) - Q[action, state])

                    rewards[episode] += np.pow(self.mdp.discount, step) * reward

                    state = s_next

                for s in range(self.mdp.nStates):
                    policy[s] = np.argmax(Q[:, s])
                
            avg_rewards += rewards
            
        avg_rewards = avg_rewards / 100

        return [Q, policy, avg_rewards] 
    
    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''
        基於模型的強化學習，使用epsilon-greedy探索。
        此處採用價值迭代，其效果應與策略迭代效果一致

        輸入:
        s0 -- 初始狀態
        defaultT -- 當某一狀態-動作對尚未被拜訪時使用的預設轉移函數
        initialR -- 獎勵函數的初始估計
        nEpisodes -- 執行回合數
        nSteps -- 每回合的執行步數
        epsilon -- 探索率


        輸出:
        V -- 價值函數: V[S]
        policy -- 最終策略
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        pred_mdp = MDP.MDP(defaultT,initialR, self.mdp.discount)
        avg_rewards = np.zeros(nEpisodes)

        for trial in tqdm(range(100)):
            V = np.zeros(self.mdp.nStates)
            policy = np.zeros(self.mdp.nStates, int)

            pred_mdp.T = defaultT.copy()
            pred_mdp.R = initialR.copy()

            counts = np.zeros([self.mdp.nActions,self.mdp.nStates])
            countsWithTransition = np.zeros([self.mdp.nActions,self.mdp.nStates,self.mdp.nStates])

            rewards = np.zeros(nEpisodes)

            for episode in range(nEpisodes):
                state = s0

                for step in range(nSteps):
                    p = np.random.random()
                    if p < epsilon:
                        action = np.random.randint(self.mdp.nActions)
                    else:
                        action = policy[state]

                    reward, new_state = self.sampleRewardAndNextState(state, action)
                    rewards[episode] += np.pow(self.mdp.discount, step) * reward

                    counts[action, state] += 1
                    countsWithTransition[action, state, new_state] += 1

                    for k in range(self.mdp.nStates):
                        pred_mdp.T[action, state, k] = countsWithTransition[action, state, k] / counts[action, state]

                    pred_mdp.R[action, state] = (pred_mdp.R[action, state] * (counts[action, state] - 1) + reward) / counts[action, state]

                    state = new_state
                
                V, _, _ = pred_mdp.valueIteration(V,nIterations=np.inf,tolerance=0.01)
                policy = pred_mdp.extractPolicy(V)
                    
            avg_rewards += rewards

        avg_rewards = avg_rewards / 100

        return [V, policy, avg_rewards]