import torch
import gym

class DictBuffer:
    """ Buffer of observations, in which the observations can be (recursive) dictionaries. Used for Dict observation spaces
    """
    def __init__(self, observation_space, args):
        self.is_dict = isinstance(observation_space, gym.spaces.Dict)

        if self.is_dict:
            self.buffers = {
                key: DictBuffer(subspace, args) for key, subspace in observation_space.items()
            }
        else:
            self.data = torch.zeros((args.erpoolsize,) + observation_space.shape)

    def put(self, index, obs):
        if self.is_dict:
            for key, buf in self.buffers.items():
                buf.put(index, obs[key])
        else:
            self.data[index] = obs

    def get(self, index):
        if self.is_dict:
            return {
                key: buf.get(index) for key, buf in self.buffers.items()
            }
        else:
            return self.data[index]

class ExperienceBuffer:
    def __init__(self, observation_space, action_space, args):
        self.args = args

        N = args.erpoolsize
        obs_shape = observation_space.shape

        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)

        if self.is_discrete:
            num_actions = action_space.n

            self.actions = torch.zeros((N,), dtype=torch.long)
            self.params = torch.zeros((N, num_actions))
            self.next_params = torch.zeros((N, num_actions))
        else:
            act_shape = action_space.shape

            self.actions = torch.zeros((N,) + act_shape)
            self.params = torch.zeros((N, 2) + act_shape)
            self.next_params = torch.zeros((N, 2) + act_shape)

        self.states = DictBuffer(observation_space, args)
        self.rewards = torch.zeros((N, 1))
        self.not_dones = torch.zeros((N, 1))
        self.next_states = DictBuffer(observation_space, args)
        self.er_index = 0
        self.er_count = 0

    def add(self, state, action, dist, reward, done, next_state, next_dist):
        self.states.put(self.er_index, state)
        self.next_states.put(self.er_index, next_state)

        if self.is_discrete:
            assert(len(dist.logits.shape) == 2)
            self.params[self.er_index] = dist.logits[0]
            self.next_params[self.er_index] = next_dist.logits[0]
        else:
            assert(len(dist.loc.shape) >= 2)
            self.params[self.er_index, 0] = dist.loc[0]
            self.params[self.er_index, 1] = dist.scale[0]
            self.next_params[self.er_index, 0] = next_dist.loc
            self.next_params[self.er_index, 1] = next_dist.scale

        self.actions[self.er_index] = action
        self.rewards[self.er_index, 0] = reward
        self.not_dones[self.er_index, 0] = float(not done)

        # TODO: Randomize er_index when the buffer is full
        self.er_index = (self.er_index + 1) % self.args.erpoolsize
        self.er_count = min(self.args.erpoolsize, self.er_count + 1)

    def num_slices(self, batch_size):
        self.indexes = torch.arange(self.er_count).split(batch_size)

        return len(self.indexes)

    def get(self, slice_index):
        """ Return states, actions, loc, scale, rewards, not_dones, next_states as arrays ready for building a TensorDataset.
            The arrays are sliced in a way that they only contain valid transitions
        """
        indexes = self.indexes[slice_index]

        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=self.params[indexes])
            next_dist = torch.distributions.Categorical(logits=self.next_params[indexes])
        else:
            dist = torch.distributions.Normal(
                loc=self.params[indexes, 0],
                scale=self.params[indexes, 1]
            )
            next_dist = torch.distributions.Normal(
                loc=self.next_params[indexes, 0],
                scale=self.next_params[indexes, 1]
            )

        return \
            self.states.get(indexes), \
            self.actions[indexes], \
            dist, \
            self.rewards[indexes], \
            self.not_dones[indexes], \
            self.next_states.get(indexes), \
            next_dist
