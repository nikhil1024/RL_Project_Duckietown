# RL_Project

The aim of this project was to teach a self-driving agent to drive around without colliding into obstacles. The agent also must learn to stay in its lane. The environment chosen was the Duckietown environment. It is a wrapper around the OpenAI Gym environment. Here is the link to the environment: https://github.com/duckietown/gym-duckietown.

We experimented with 3 algorithms:
- Deep Q Networks (DQN) for discrete control
- Deep Deterministic Policy Gradients (DDPG) for continuous control
- Monte Carlo Tree Search (MCTS) + Rollouts for discrete control

Discrete control only allowed for three different movements: go left, go straight, or go right.

We achieved decent results for all 3 algorithms, achieving a moving average reward well over 1000 in all 3 algorithms allowing just 250 time steps per episode.

DQN inherently suffers from a lot of drawbacks such as high gradient variance, and also only being restricted to a discrete action space may not be ideal for a self-driving car.

DDPG improves over its drawbacks by introducing an actor-critic like architecture. DDPG can also take advantage of a continuous action space. Thus, it is no surprise that it performs much better than DQN.

MCTS, which is also restricted to a discrete action space, does not suffer from other drawbacks of DQN and is able to achieve comparable results to DDPG just because of the sheer power of the algorithm. MCTS takes advantage of a tree search, which we combine with a random Rollout strategy and is able to search some crucial parts of the search space leading to a quite impressive performance.

In conclusion, we found that MCTS ~ DDPG >>> DQN for a self-driving agent in the Duckietown environment.
