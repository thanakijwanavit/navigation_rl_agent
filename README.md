[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Task
Build and train an agent to collect bananas in the state space with an input of 37 dimensions in
[Unity's Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector)

The goal is to achieve an average score of +13 over 100 episodes

## Approach

### 1. Evaluate state and action spaces
state is an array of 37 float values, where numbers represent variables include velocity, preception of objects, and colour of precieved object

action space is an integer range from 0 to 4 as described in the introduction section

### 2. Benchmark
run an agent which make decisions at random in order to establish a baseline benchmark
```
env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score)) 
```
This agent seems to give a score between -1 and 1. it will never get 13 which is the require score to consider the task solved

### 3. Construct Q agent and alternative algorithms

Deep Q agent uses a policy to decide on action. The backend of deep Q is a neural network,
which is trained by simulation over the game space

### Q and reward function
A simple Q function is used to generate a reward for each possible state and that reward is used as policy for the agent to choose action from.

Reward = Q(s,a)

There are various Q functions that are well researched and has good proven track record
However, deep Q network is the one chosen due to the success rate.

### Deep Q function
deep Q function uses a neural network to estimate the action score. This is stored in model.py consists of 2 hidden layers of 64 default node in both hidden layers.

experience replay is used to feed input sequential batches of tuples into the network. The experience is stored as a class ReplayBuffer implemented in dqn_agent.py. Experience is stored after every pass of network decision step

### 4. optimize hyperparameters

5 values of fp1 and fp2 has been tested and the number of nodes required has been plotted.
These are fp1 = fp2 = [16,32,64,128,256]


The results are as follows
![fp16](https://github.com/thanakijwanavit/navigation_rl_agent/blob/master/fp16.png?raw=true)
![fp32](https://github.com/thanakijwanavit/navigation_rl_agent/blob/master/fp32.png?raw=true)
![fp64](https://github.com/thanakijwanavit/navigation_rl_agent/blob/master/fp64.png?raw=true)
![fp128](https://github.com/thanakijwanavit/navigation_rl_agent/blob/master/fp128.png?raw=true)
![fp256](https://github.com/thanakijwanavit/navigation_rl_agent/blob/master/fp256.png?raw=true)


### 5. Best performing agent 

The best performing agent was DQN with Experience replay using network of fp1=fp2 = 64


# Evaluation and improvement

Many other improvement can be made if time and resources becomes available
1. Double Deep Q network
this prevents some extremely high reward value predicted from inexperienced network. This is implemented by using one set of Q to determine the best action and another set to evaluate that action.
2. Dueling Agents
This use 2 streams of network one to estimate state and the other one to estimate advantage for the action. The output of both are then used to calculate the Q values
The reasoning for this is that the state don't change much across actions however, we need to measure impact of each action hence the advantage function.

3. Prioritized experience replay
Experience are selected based on a priority value based on magnitude of an error. This will be useful in case a rare state happens


# Project started instruction

1.Download environment from this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
2.run ``` git clone https://github.com/thanakijwanavit/navigation_rl_agent.git```
3. ```cd navigation_rl_agent```
4. ``` pip install -r requirements.txt```
5. start jupyter notebook and read the file Navigation.ipynb