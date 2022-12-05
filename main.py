import gym
import argparse

from qvalueweightedregression import QWR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Q-Value Weighted Regression for Reinforcement Learning")

    parser.add_argument("--env", type=str, required=True, help="Environment name (Gym environment)")
    parser.add_argument("--augment", type=int, default=0, help="Use state augmentation when forwarding neural networks (shifting images for instance)")

    parser.add_argument("--hidden", type=int, default=64, help="Number of hidden layers in the policy, guide and critic")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers in the policy, guide and critic")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the neural networks")

    parser.add_argument("--erpoolsize", type=int, default=100000, help="Experience buffer size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for each learning epoch")
    parser.add_argument("--erfreq", type=int, default=1, help="How often to do learning")

    parser.add_argument("--action_samples", type=int, default=8, help="Number of actions sampled when computing argmaxes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature of the log-sum-exp operation used to compute target Q-Values")
    parser.add_argument("--beta", type=float, default=1.0, help="Temperature used when computing dzeta values (exp(Advantage / beta))")

    args = parser.parse_args()

    env = gym.make(args.env)
    learner = QWR(env, args)
    learner.learn()
