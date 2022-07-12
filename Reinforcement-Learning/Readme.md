# Introduction
Reinforcement learning involves an agent, a set of states, and a set of actions per state.
By performing an action, the agent transitions from state to state. Executing an action in a specific state provides the agent with a reward (a numerical score).
## Q-learning
### Dependency
`pip install gym`  
`pip install ipympl`  
`pip install pygame`  
### Algorithm
$$Q(s,a) \leftarrow Q(s,a)+\alpha[r(t)+\gamma \max(Q(s_{t+1},a_{t+1}))-Q(s,a)]$$  
