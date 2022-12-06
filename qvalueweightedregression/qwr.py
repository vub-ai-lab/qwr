import torch
import gym

import sys
import random

from tqdm import tqdm

from .models import Actor, Critic
from .experience_buffer import ExperienceBuffer

class QWR:
    def __init__(self, env, args):
        # Wrap the environment if needed
        if 'Atari' in str(env): # For instance, "<OrderEnforcing<AtariEnv<ALE/Pong-v5>>>"
            import stable_baselines3 as sb3

            env = sb3.common.atari_wrappers.AtariWrapper(env)
            print('Wrapping Atari env')

        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.args = args
        self.env = env
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # Make 2 neural networks: the critic Q(s, a) -> R, and the actor pi(s) -> Normal
        self.actor = Actor(env.observation_space, env.action_space, args)
        self.critic = Critic(env.observation_space, env.action_space, args)
        self.target = Critic(env.observation_space, env.action_space, args)

        # Experience buffer
        self.buffer = ExperienceBuffer(env.observation_space, env.action_space, args)

        # Load the weights of the actor and critic?
        if args.load is not None:
            print('Loading', args.load)

            aw, cw = torch.load(args.load)
            self.actor.load_state_dict(aw)
            self.critic.load_state_dict(cw)

        self.target.copy(self.critic)

    def get_qvalue_samples(self, s, dist, n_samples):
        repeats = [1] * len(s.shape)
        repeats[0] = n_samples

        repeated_states = torch.tile(s, tuple(repeats))

        actions = self.actor.get_actions(dist, sample_shape=(n_samples,))               # Shape (n_samples, batch_size, *action_shape)
        actions = torch.flatten(actions, start_dim=0, end_dim=1)                        # Shape (batch_size * n_samples, *action_shape)
        qvalues = self.target.get_qvalues(repeated_states, actions)                     # Shape (batch_size * n_samples, 1)
        qvalues = qvalues.reshape(n_samples, -1)                                        # Shape (n_samples, batch_size)

        return qvalues, actions

    def slices_generator(self, num_slices):
        return tqdm(sorted(range(num_slices), key=lambda x: random.random()))

    def train_epoch(self):
        n_samples = self.args.action_samples
        num_slices = self.buffer.num_slices(self.args.batch_size)

        # Train the critic
        for slice_index in self.slices_generator(num_slices):
            states, actions, dist, rewards, not_dones, next_states, next_dist = self.buffer.get(slice_index)

            with torch.no_grad():
                # Target values based on Q(s', a' ~ next_dist)
                next_qvalues, _ = self.get_qvalue_samples(next_states, next_dist, n_samples)    # Shape (n_samples, batch_size)

                # Top-K
                topk = torch.topk(next_qvalues, self.args.topk, dim=0, sorted=False).values     # Shape (k, batch_size)
                next_qvalues = topk.mean(0)                                                     # Shape (batch_size,)

                target_qvalues = rewards.flatten() + not_dones.flatten() * self.args.gamma * next_qvalues

            current_qvalues = self.critic.get_qvalues(states, actions).flatten()
            loss = torch.mean((current_qvalues - target_qvalues) ** 2)
            print('QL', loss.item(), current_qvalues[0].item(), target_qvalues[0].item())

            self.critic.optimizer().zero_grad()
            loss.backward()
            self.critic.optimizer().step()

        self.target.copy(self.critic)

        # Train the actor
        for slice_index in self.slices_generator(num_slices):
            states, actions, dist, rewards, not_dones, next_states, next_dist = self.buffer.get(slice_index)

            with torch.no_grad():
                # State values based on the average of Q(s, a ~ dist)
                qvalues, sampled_actions = self.get_qvalue_samples(states, dist, n_samples)                   # Shape (n_samples, batch_size)
                values = qvalues.mean(0, keepdim=True)                                                   # Shape (1, batch_size)

                advantages = qvalues - values                       # Shape (n_samples, batch_size)
                dzetas = torch.exp(advantages / self.args.beta)     # Shape (n_samples, batch_size)
                dzetas = torch.clip(dzetas, 1e-2, 20.0)

                inverted_sampled_actions = self.actor.invert_actions(sampled_actions)                    # Shape (batch_size * n_samples, *action_shape)
                inverted_sampled_actions = torch.unflatten(inverted_sampled_actions, 0, (n_samples, -1)) # Shape (n_samples, batch_size, *action_shape)

            log_times_dzeta = []
            current_dist = self.actor.get_dist(states)

            for i in range(n_samples):
                sl = inverted_sampled_actions[i, ...]               # Shape (batch_size, *action_shape)
                log_prob = current_dist.log_prob(sl)

                if not self.is_discrete:
                    log_prob = log_prob.sum(tuple(range(1, len(sl.shape))))    # Shape (batch_size,)

                dzeta = dzetas[i]                                   # Shape (batch_size,)
                log_times_dzeta.append(log_prob * dzeta)            # Shape (batch_size,)

            loss = -torch.mean(torch.cat(log_times_dzeta))

            if loss.isnan().any() or loss.isinf().any():
                continue

            print('AL', loss.item())

            self.actor.optimizer().zero_grad()
            loss.backward()
            self.actor.optimizer().step()

    def learn(self):
        ep_number = 0
        ts_number = 0

        while True:
            state = torch.from_numpy(self.env.reset())
            done = False
            ret = 0.0
            ep_number += 1

            while not done:
                with torch.no_grad():
                    dist = self.actor.get_dist(state[None, ...])
                    action = self.actor.get_actions(dist)[0]

                next_state, reward, done, info = self.env.step(action.numpy())
                next_state = torch.from_numpy(next_state)

                # Add the experience in the buffer
                with torch.no_grad():
                    next_dist = self.actor.get_dist(next_state[None, ...])

                self.buffer.add(state, action, dist, reward, done, next_state, next_dist)

                ret += reward
                ts_number += 1
                state = next_state

                if (self.args.max_timesteps is not None) and ts_number >= self.args.max_timesteps:
                    return

                # Perform learning if now is the time for it
                if (ts_number % self.args.erfreq) == 0 and ts_number > 1000:
                    self.train_epoch()

                    # Save the latest version of the actor and critic weights
                    if self.args.save is not None:
                        aw = self.actor.state_dict()
                        cw = self.critic.state_dict()

                        torch.save((aw, cw), self.args.save)

            print('R', ep_number, ts_number, ret)
            sys.stdout.flush()

            if (self.args.max_episodes is not None) and ep_number >= self.args.max_episodes:
                return
