## Part 1
- Test your code for model-based RL with the maze problem described in TestRL2Maze.py (same maze problem as in Assignment 1). Produce a graph where the x-axis indicates the episode # (from 0 to 200) and the y-axis indicates the average (based on 100 trials) of the cumulative discounted rewards per episode (100 steps) when each episode starts in state 0. The graph should contain 2 curves corresponding to [1 point]
  - Model-based RL (epsilon=0.05, the default transition function is uniform when a state-action pair has not been visited and the initial expected reward is 0 for all state-action pairs)
  - Q-learning (epsilon=0.05 and the initial Q-function is 0 for all state-action pairs)
![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_1/Figure_2.png)
- Explain the results. Discuss how different properties of each algorithm influence the cumulative discounted rewards per episode earned during training as well as the resulting Q-values and policy. [1 point]
  - 結果
    - Model-based RL與Q-Learning在經過約略50個Episode後皆能達到平穩，並落於約為50的平均獎勵
    - Model-based收斂速度快於Q-Learning
  - 分析
    - Model-based能充分利用轉移模型與獎勵模型，其維度比Q-Table更高，能夠更準確表達環境
    - 此外由於迷宮模型比較簡單，轉移模型可以快速接近真實狀態
    - 該因素導致Model-based僅需較少次嘗試即可達到較好的結果
  - Model-based RL
    - 如果一開始就具有對模型的初步評估，且模型的建構成本較低，使用Model-based RL可以在初期快速取得成果
    - 越穩定、簡單的環境，Model-based RL的表現越好
  - Q-Learning
    - 僅需維護Q-Table，在成本上具有優勢
    - 可遷移性較好，且不用先評估模型
    - 對於複雜環境的處理能力較優
- Test your code for UCB, epsilon-greedy bandit and Thompson sampling with the simple multi-armed bandit problem described in TestBandit.py. Produce a graph where the x-axis indicates the iteration # (from 0 to 200) and the y-axis indicates the average (based on 1000 trials) of the reward earned at each iteration. The graph should contain 3 curves corresponding to 
  - UCB [0.5 point]
  - Epsilon-greedy bandit (epsilon = 1 / # iterations) [0.5 point]
  - Thompson sampling (k=1 and prior consists of Beta distributions with all hyper parameters set to 1) [1 point]
![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_1/Figure_1.png)
- Explain the results. Are the results surprising? Do they match what is expected based on the theory? [1 point]
  - 結果
    - UCB與Thompson Sampling於25 Episodes即具有0.6的平均獎勵，其於150 Episodes時收斂於0.68的平均獎勵
    - Epsilon-greedy於25 Episode僅有0.55的平均獎勵，其於80 Episodes時收斂於0.6的平均獎勵
  - 分析
    - UCB與Thompson Sampling具有相似的結果，兩者皆優於Epsilon-greedy策略
    - 從數學上來說，UCB與Thompson Sampling都為對數收斂，而Epsilon-greedy在合適decay也為對數收斂，但UCB具有較小的係數
      - [UCB & Epsilon-greedy的Regret證明出處](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf) 
    - 從直覺來看，UCB與Thompson Sampling的探索是有傾向的，而Epsilon的探索更接近於盲目選擇，結果符合預期

## Part 2
- Produce 3 graphs that show the performance of REINFORCE, REINFORCE with baseline and PPO on the cartpole problem. Running "python REINFORCE.py --mode=cartpole" produces the graph for REINFORCE which is saved in the "images" directory. For each graph, the y axis corresponds to the cumulative rewards (averaged over the last 25 episodes) and the x axis is the number of episodes (up to 800 for REINFORCE and REINFORCE with baseline, and up to 150 for PPO). [2 points] Explain the results based on the properties of each algorithm [1 point].
  - 結果
    - REINFORCE

      ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/1-1.png)
      - 於Episode = 450時開始有效提升，至Episode = 700時趨於平穩並訓練至獎勵接近200
      - 波動較大
    - REINFORCE with baseline

      ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/1-2.png)
      - 於Episode = 400時開始有效提升，至Episode = 650時趨於平穩並訓練至獎勵接近200
      - 波動較小
    - PPO

      ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/1-3.png)
      - 於Episode = 80時開始有效提升，至Episode = 100時趨於平穩並訓練至獎勵接近200
      - 波動最小，訓練曲線較為平緩
  - 分析
    - REINFORCE添加baseline後，減少了方差，因此避免了部分的錯誤更新，且也較少在訓練平穩後仍然產生一定幅度的震盪
    - PPO採用clipped objective限制了更新的幅度，在三次實驗中方差最小
    - PPO透過樣本重用，大幅提升收斂速度，同時也使得平行處理的效率較好，在訓練時間上也為最快
- (Optional) What happens if you change POLICY_TRAIN_ITERS from 1 to 10 in REINFORCE with baseline? Explain your observations. [1 bonus point]
  - 結果
 
    ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/1-4.png)
    - 相對於POLICY_TRAIN_ITERS = 1，POLICY_TRAIN_ITERS = 10的收斂速度較快
    - 於Episode = 0時開始有效提升，至Episode = 250時趨於平穩並訓練至獎勵接近200
    - 但在Episode = 700時，模型嚴重失效，其獎勵降至約10，且後續無法繼續訓練
  - 分析
    - 透過樣本重用，策略網路以非常激進的幅度進行更新，這使得其提升速度大幅增加
    - 後期模型的嚴重失效為過擬合，其對每次環境變化的敏感度超過了合理範圍，導致後續再無改善機會
    - REINFORCE不適合樣本重用，其為on-policy演算法，當重用樣本時，樣本與模型不再匹配
    - 此現象在試圖將Q-Learning常用的Replay Buffer應用在REINFORCE也可見同現象，當前樣本只應符合於當前模型，試圖用非當前樣本更新當前模型都會導致更新方向的錯誤

- Produce 3 graphs that show the performance of REINFORCE, REINFORCE with baseline and PPO on the mountain-car problem. Read a brief description of the Mountain-Car problem from Open AI Gym. The goal is to move a car up a mountain. The mountain-car environment has a reward as follows: each time step it gives a reward of -1, until the goal is reached (at a certain height of the mountain where the reward is 0) or the episode reaches termination (in 200 steps). Running "python REINFORCE.py --mode=mountain_car" produces the graph for REINFORCE which is saved in the "images" directory. For each graph, the y axis corresponds to the cumulative rewards (averaged over the last 25 episodes) and the x axis is the number of episodes. What do you notice about the performance of the algorithms on mountain-car versus cartpole? What is a possible explanation for the performance differences in terms of the details of the environments? [1 point]
  - 結果

    ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/2-1.png)
    ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/2-2.png)
    ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/2-3.png)
    - 無論哪種演算法都無法有效訓練
  - 分析
    - 其原因來自訓練時獎勵的不合理設置，在Mountain-Car中，獎勵僅當到達山頂時為0，其餘時間皆為-1
    - 其導致前期所有的訓練結果獎勵皆為0，模型無從判別何種動作是正確的
    - 任何演算法只要沒有能力在一開始就到達山頂，就必然不可能進行訓練
- Produce 3 graphs that show the performance of REINFORCE, REINFORCE with baseline and PPO on a modified version of the mountain problem. This environment is the same as plain mountain-car, except that the reward function is changed so that the reward at each time step is the height of the car. The goal is the same as plain mountain-car (i.e., to reach the top of the mountain). Running "python REINFORCE.py --mode=mountain_car_mod" produces the graph for REINFORCE which is saved in the "images" directory. For each graph, the y axis corresponds to the cumulative rewards (averaged over the last 25 episodes) and the x axis is the number of episodes. Compare the performances of the algorithms for mountain-car versus the modified mountain-car. Specifically, compare the reward curves and the final height achieved by the agents. Explain your observations. [1 point]
  - 結果
    - REINFORCE

      ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/3-1.png)
      - 訓練效果較差，但仍然有限度的在逐漸提升，Episode = 700時獎勵約為-99
      - 波動非常大
    - REINFORCE with baseline

      ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/3-2.png)
      - 訓練效果良好，雖然沒有CartPole那種迅速增長但整體趨勢明顯，至Episode = 600時趨於平穩並訓練至獎勵接近-78
      - 波動較小
    - PPO

      ![image](https://github.com/newyen2/1141_P75H400_HW2/blob/main/Part_2/images/3-3.png)
      - 於Episode = 60時開始有效提升，至Episode = 100時趨於平穩並訓練至獎勵接近-75
      - 波動最小，訓練曲線較為平緩
  - 分析
    - 重新設計獎勵後，於三個演算法都能發現明顯進步，模型能夠從訓練中得到一定改進資訊
    - REINFORCE明顯不適應Mountain-Car，由於方差極大，且錯誤的更新對於該環境會造成非常嚴重的退步
    - REINFORCE with baseline在該環境明顯體現出其與REINFORCE的方差影響
    - PPO也能良好適應該環境，不過其與REINFORCE的收斂結果相近，應有更好的獎勵設計來達成更好的獎勵結果
