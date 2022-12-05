# Q-Value Weighted Regression

Paper: [Q-Value Weighted Regression: Reinforcement Learning with Limited Data](https://ieeexplore.ieee.org/abstract/document/9892633?casa_token=7Br-eD7yKp8AAAAA:h2dLYzSXlXv-e9y6VnIrPFoKXWPh2eo_htIUwGPV-WFinCMVxPlsWw0jSzoZvOrDHWVFc-DThgO9aA)

Q-Value Weighted Regression is a relatively simple RL algorithm that trains a stochastic policy so that the probability of every action increases, with a force (weight) proportional to the advantage value of that action. The advantage value of an action is computed as $A(s, a) = Q(s, a) - E[Q(s, a')]$.

This repo implements QwR as I understand it from the paper (that releases no code). With limited hyper-parameter tuning, the code in this repository learns LunarLander and LunarLanderContinuous. It also runs on Pong but does not seem to learn.

## Features

- Interacts with an OpenAI Gym environment
- Support for discrete and continuous action spaces (Discrete and Box spaces)
- Support for Discrete, Box and Dict observation spaces. Images are fed through a NatureCNN.
- Simple logging: stuff gets printed on stdout with a prefix, for ease of use with gnuplot.
- Simple code without advanced features, designed to quickly experiment with the algorithm.
