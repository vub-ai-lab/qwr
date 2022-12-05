import torch
import numpy as np
import gym

from .observation_extractors import make_extractor

class Function(torch.nn.Module):
    """ Neural network that starts with an observation extractor (see observation_extractors.py)
        then a few hidden layers, to finally product an output of a given shape
    """
    def __init__(self, observation_space, output_shape, args):
        super().__init__()

        output_len = int(np.prod(np.array(output_shape)))

        # Extractor
        extractor = make_extractor(observation_space, args)

        # Other layers
        layers = [extractor]
        layers.append(torch.nn.Linear(extractor.output_len, args.hidden))
        layers.append(torch.nn.ReLU())

        for i in range(args.layers-1):
            layers.append(torch.nn.Linear(args.hidden, args.hidden))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(args.hidden, output_len))
        layers.append(torch.nn.Unflatten(1, output_shape))

        self.fwd = torch.nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.fwd.parameters(), lr=args.lr)

    def forward(self, x):
        return self.fwd(x)

class Critic:
    """ Critic that maps an observation and action to a Q-Value
    """

    def __init__(self, observation_space, action_space, args):
        # Discrete or continuous actions
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        if self.is_discrete:
            # Produce one Q-Value per action
            self.function = Function(observation_space, (action_space.n,), args)
        else:
            # Take the action as parameter and produce its Q-Value
            observation_space = gym.spaces.Dict({
                'obs': observation_space,
                'act': action_space
            })
            self.function = Function(observation_space, (1,), args)

        self.optimizer = self.function.optimizer

    def get_qvalues(self, obs, act):
        if self.is_discrete:
            all_qvalues = self.function(obs)    # Q-Values for every action in every state
            return all_qvalues.gather(dim=1, index=act[:, None])
        else:
            x = {
                'obs': obs,
                'act': act
            }

            return self.function(x)

    def copy(self, other):
        self.function.load_state_dict(other.function.state_dict())

class Actor:
    """ Stochastic actor that maps an observation to the parameters of a distribution over actions
    """
    def __init__(self, observation_space, action_space, args):
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        if self.is_discrete:
            # Produce logits
            self.function = Function(observation_space, (action_space.n,), args)
        else:
            # Produce means and std
            self.function = Function(observation_space, (2,) + action_space.shape, args)

            self.mid = torch.from_numpy((action_space.low + action_space.high) / 2.)
            self.scale = torch.from_numpy((action_space.high - action_space.low) / 2.)

        self.optimizer = self.function.optimizer

    def get_dist(self, obs):
        if self.is_discrete:
            logits = self.function(obs)
            return torch.distributions.Categorical(logits=logits)
        else:
            meanstd = self.function(obs)
            mean = meanstd[:, 0, :]
            log_std = meanstd[:, 1, :]
            std = torch.exp(torch.clamp(log_std, -3.0, 2.0))

            return torch.distributions.Normal(mean, std)

    def get_actions(self, dist, sample_shape=torch.Size([])):
        a = dist.sample(sample_shape)

        if self.is_discrete:
            # No need to adjust the action, it is already a batch of integers
            return a
        else:
            a = torch.tanh(a)       # Go to [-1, 1]
            a = a * self.scale      # Go to [-scale, scale]
            a = a + self.mid        # Move to the actual center of the action distribution
            return a

    def invert_actions(self, actions):
        """ Compute the inverse of get_actions
        """
        if self.is_discrete:
            return actions
        else:
            a = actions - self.mid
            a = a / self.scale
            a = torch.atanh(a)
            return a
