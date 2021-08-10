from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
import random


class EnvReplayBufferSequence(SimpleReplayBuffer):
    def __init__(
            self,
            max_ep_num,
            env,
            fixed_sequence,
            fixed_sequence_use_prediction,
            env_info_sizes=None
    ):
        self.max_ep_num = max_ep_num
        self.fixed_sequence = fixed_sequence
        self.fixed_sequence_use_prediction = fixed_sequence_use_prediction
        from collections import deque
        self.ep_buffer = deque(maxlen=max_ep_num)
        self.max_ep_len = 300

        self.env = env
        self._action_space = env.action_space
        self._ob_space = env.observation_space
        self.obs_to_image_obs = env.obs_to_image_obs
        self.t = 1000
        self.max_sampled_ep_len = 0
        self.use_min = False

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=1,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )
        self._max_replay_buffer_size = max_ep_num

    def add_paths(self, paths):
        for path in paths:
            path_len = len(path['actions'])
            partition = path_len // self.fixed_sequence + 1
            # print(path_len, partition)
            # print(path.keys())
            obs = path['observations']
            a = path['actions']
            r = path['rewards']
            t = path['terminals']
            next_obs = path['next_observations']

            for i in range(partition):
                if i+1 == partition:
                    seq_len = path_len - i * self.fixed_sequence
                else:
                    seq_len = self.fixed_sequence
                obs_tmp = []
                a_tmp = []
                r_tmp = []
                t_tmp = []
                next_obs_tmp = []
                for j in range(seq_len):
                    index = i * self.fixed_sequence + j
                    a_tmp.append(a[index])
                    r_tmp.append(r[index])
                    t_tmp.append(t[index])
                    # print(index, obs[index].shape, next_obs[index].shape)
                    if j == 0:
                        obs_tmp.append(obs[index][:self.env.a2])
                        # obs[index] is used since this part only executes when j=0
                        # which safely picks status at obs[i*self.fixed_sequence]
                        if self.fixed_sequence_use_prediction:
                            if i == 0:
                                init_hidden = obs[index][self.env.vision_index:self.env.a2]
                            else:
                                init_hidden = obs[index][self.env.a2:self.env.b2]
                        else:
                            init_hidden = obs[index][self.env.vision_index:self.env.a2]
                    else:
                        o = np.zeros(self.env.b2 + j * self.env.i2)
                        o[:self.env.a2] = obs[index][:self.env.a2]
                        o[self.env.a2:self.env.b2] = init_hidden
                        o[self.env.b2:] = obs[index][self.env.b2 + i * self.fixed_sequence * self.env.i2:]
                        obs_tmp.append(o)
                    next_o = np.zeros(self.env.b2 + (j + 1) * self.env.i2)
                    next_o[:self.env.a2] = next_obs[index][:self.env.a2]
                    next_o[self.env.a2:self.env.b2] = init_hidden
                    next_o[self.env.b2:] = next_obs[index][self.env.b2 + i * self.fixed_sequence * self.env.i2:]
                    next_obs_tmp.append(next_o)

                sub_path = dict(observations=obs_tmp,
                                actions=a_tmp,
                                rewards=r_tmp,
                                next_observations=next_obs_tmp,
                                terminals=t_tmp,)
                self.ep_buffer.append((sub_path, seq_len))
                self._advance()

    def random_batch(self, batch_size):
        if self.t >= self.max_sampled_ep_len:
            self.sampled_ind = np.random.choice(a=self._size, size=batch_size)
            self.sampled_ep = []
            self.sampled_ep_len = []
            for ind in self.sampled_ind:
                self.sampled_ep.append(self.ep_buffer[ind][0])
                self.sampled_ep_len.append(self.ep_buffer[ind][1])
            self.sampled_ep_len = np.array(self.sampled_ep_len)
            self.t = 0
            if self.use_min:
                self.max_sampled_ep_len = np.min(self.sampled_ep_len)
            else:
                self.max_sampled_ep_len = np.max(self.sampled_ep_len)

        if self.use_min:
            batch_ep = self.sampled_ep
            new_batch_size = batch_size
        else:
            batch_ep_index = np.argwhere(self.sampled_ep_len > self.t).flatten()
            batch_ep = [self.sampled_ep[ind] for ind in batch_ep_index]
            new_batch_size = len(batch_ep)

        if self.t == 0:
            observations = np.zeros((new_batch_size, self.env.a1), np.float32)
        else:
            observations = np.zeros((new_batch_size, self.env.b1 + self.t * self.env.i1), np.float32)

        actions = np.zeros((new_batch_size, 3), np.float32)
        rewards = np.zeros((new_batch_size, 1), np.float32)
        terminals = np.zeros((new_batch_size, 1), np.float32)
        next_observations = np.zeros((new_batch_size, self.env.b1 + (self.t+1) * self.env.i1), np.float32)

        for i in range(new_batch_size):
            ep = batch_ep[i]
            observations[i] = self.obs_to_image_obs(ep['observations'][self.t])
            a_onehot = np.zeros((3,), np.float32)
            a_onehot[ep['actions'][self.t]] = 1
            actions[i] = a_onehot
            rewards[i] = ep['rewards'][self.t]
            terminals[i] = ep['terminals'][self.t]
            next_observations[i] = self.obs_to_image_obs(ep['next_observations'][self.t])

        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_observations=next_observations,
        )
        self.t += 1
        return batch


class EnvReplayBufferRandom(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.ob_dim_true = env.ob_dim_true
        self.obs_to_image_obs = env.obs_to_image_obs

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if len(observation) == get_dim(self._ob_space):
            if isinstance(self._action_space, Discrete):
                new_action = np.zeros(self._action_dim)
                new_action[action] = 1
            else:
                new_action = action
            return super().add_sample(
                observation=observation,
                action=new_action,
                reward=reward,
                next_observation=next_observation,
                terminal=terminal,
                **kwargs
            )

    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)
        obs = batch['observations']
        next_obs = batch['next_observations']
        img_obs = np.zeros((batch_size, self.ob_dim_true), np.float32)
        next_img_obs = np.zeros((batch_size, self.ob_dim_true), np.float32)

        for i in range(batch_size):
            img_obs[i] = self.obs_to_image_obs(obs[i])
            next_img_obs[i] = self.obs_to_image_obs(next_obs[i])

        batch['observations'] = img_obs
        batch['next_observations'] = next_img_obs
        return batch


class EnvReplayBufferEpReal(SimpleReplayBuffer):
    def __init__(
            self,
            max_ep_num,
            env,
            env_info_sizes=None
    ):
        from env_survive.env_survive_real import T, pred_status
        self.max_ep_num = max_ep_num
        from collections import deque
        self.ep_buffer = deque(maxlen=max_ep_num)
        # self.ep_len_buffer = deque(maxlen=max_ep_num)
        self.max_ep_len = 300
        self.bptt = T
        self.s_len = 784 + 2 + pred_status

        self.env = env
        self._action_space = env.action_space
        self._ob_space = env.observation_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=1,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )
        self._max_replay_buffer_size = max_ep_num

    def add_paths(self, paths):
        for path in paths:
            path_len = len(path['actions'])
            if path_len > self.bptt:
                self.ep_buffer.append((path, path_len))
                # self.ep_len_buffer.append(path_len)
                self._advance()

    def random_batch(self, batch_size):
        # ep_len_dist = np.asarray(list(self.ep_len_buffer))
        # ep_len_dist = ep_len_dist / np.sum(ep_len_dist)
        sampled_ind = np.random.choice(a=self._size, size=batch_size,
                                       # p=ep_len_dist
                                       )
        sampled_ep = []
        for ind in sampled_ind:
            sampled_ep.append(self.ep_buffer[ind])
        len_sampled_ep = len(sampled_ep)
        observations = np.zeros((batch_size, self._observation_dim), np.float32)
        actions = np.zeros((batch_size, 3), np.float32)
        rewards = np.zeros((batch_size, 1), np.float32)
        terminals = np.zeros((batch_size, 1), np.float32)
        next_observations = np.zeros((batch_size, self._observation_dim), np.float32)

        for i in range(len_sampled_ep):
            ep, ep_len = sampled_ep[i]
            o = ep['observations']
            a = ep['actions']
            r = ep['rewards']
            next_o = ep['next_observations']
            term = ep['terminals']

            t = np.random.choice(a=ep_len - self.bptt, size=None) + self.bptt

            observations[i, 0:784] = o[t][0:784]
            observations[i, 784:786] = o[t][784:786]
            observations[i, 786:self.s_len] = o[t][786:self.s_len]
            for k, j in enumerate(range(t-self.bptt, t)):
                observations[i, self.s_len + 787 * k:self.s_len + 784 + 787 * k] = o[j][0:784]
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[j]] = 1
                observations[i, self.s_len + 784 + 787 * k:self.s_len + 787 + 787 * k] = a_onehot
            a_onehot = np.zeros((3,), np.float32)
            a_onehot[a[t]] = 1
            actions[i] = a_onehot
            rewards[i] = r[t]
            terminals[i] = term[t]
            next_observations[i, 0:784] = next_o[t][0:784]
            next_observations[i, 784:786] = next_o[t][784:786]
            next_observations[i, 786:self.s_len] = next_o[t][786:self.s_len]
            for k, j in enumerate(range(t-self.bptt+1, t+1)):
                next_observations[i, self.s_len + 787 * k:self.s_len + 784 + 787 * k] = o[j][0:784]
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[j]] = 1
                next_observations[i, self.s_len + 784 + 787 * k:self.s_len + 787 + 787 * k] = a_onehot
        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_observations=next_observations,
        )
        return batch


class EnvReplayBufferEp(SimpleReplayBuffer):
    def __init__(
            self,
            max_ep_num,
            env,
            env_info_sizes=None
    ):
        from env_survive.env_survive_tmp import T, pred_status
        self.max_ep_num = max_ep_num
        from collections import deque
        self.ep_buffer = deque(maxlen=max_ep_num)
        self.ep_len_buffer = deque(maxlen=max_ep_num)
        self.max_ep_len = 300
        self.bptt = T
        self.s_len = 784 + 2 + pred_status
        self.sample_different = True

        self.env = env
        self._action_space = env.action_space
        self._ob_space = env.observation_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_ep_num,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_paths(self, paths):
        for path in paths:
            path_len = len(path['actions'])
            if path_len > self.bptt:
                self.ep_buffer.append((path, path_len))
                self.ep_len_buffer.append(path_len)
                self._advance()

    def random_batch(self, batch_size):
        ep_len_dist = np.asarray(list(self.ep_len_buffer))
        ep_len_dist = ep_len_dist / np.sum(ep_len_dist)
        sampled_ind = np.random.choice(a=self._size, size=batch_size, p=ep_len_dist)
        sampled_ep = []
        for ind in sampled_ind:
            sampled_ep.append(self.ep_buffer[ind])
        len_sampled_ep = len(sampled_ep)
        observations = np.zeros((batch_size, self._observation_dim), np.float32)
        actions = np.zeros((batch_size, 3), np.float32)
        rewards = np.zeros((batch_size, 1), np.float32)
        terminals = np.zeros((batch_size, 1), np.float32)
        next_observations = np.zeros((batch_size, self._observation_dim), np.float32)

        for i in range(len_sampled_ep):
            ep, ep_len = sampled_ep[i]
            o = ep['observations']
            a = ep['actions']
            r = ep['rewards']
            next_o = ep['next_observations']
            term = ep['terminals']

            t = np.random.choice(a=ep_len - self.bptt, size=None) + self.bptt

            observations[i, 0:784] = o[t][0:784]
            observations[i, 784:786] = o[t][784:786]
            observations[i, 786:self.s_len] = o[t][786:self.s_len]
            for k, j in enumerate(range(t-self.bptt, t)):
                observations[i, self.s_len + 787 * k:self.s_len + 784 + 787 * k] = o[j][0:784]
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[j]] = 1
                observations[i, self.s_len + 784 + 787 * k:self.s_len + 787 + 787 * k] = a_onehot
            a_onehot = np.zeros((3,), np.float32)
            a_onehot[a[t]] = 1
            actions[i] = a_onehot
            rewards[i] = r[t]
            terminals[i] = term[t]
            next_observations[i, 0:784] = next_o[t][0:784]
            next_observations[i, 784:786] = next_o[t][784:786]
            next_observations[i, 786:self.s_len] = next_o[t][786:self.s_len]
            for k, j in enumerate(range(t-self.bptt+1, t+1)):
                next_observations[i, self.s_len + 787 * k:self.s_len + 784 + 787 * k] = o[j][0:784]
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[j]] = 1
                next_observations[i, self.s_len + 784 + 787 * k:self.s_len + 787 + 787 * k] = a_onehot
        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_observations=next_observations,
        )
        return batch


class EnvReplayBufferSeq_(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if len(observation) == get_dim(self._ob_space):
            if isinstance(self._action_space, Discrete):
                new_action = np.zeros(self._action_dim)
                new_action[action] = 1
            else:
                new_action = action
            return super().add_sample(
                observation=observation,
                action=new_action,
                reward=reward,
                next_observation=next_observation,
                terminal=terminal,
                **kwargs
            )


class EnvReplayBufferSeq(SimpleReplayBuffer):
    def __init__(
            self,
            max_ep_num,
            env,
            env_info_sizes=None
    ):
        from env_survive.env_survive_tmp import T, pred_status
        self.max_ep_num = max_ep_num
        from collections import deque
        self.ep_buffer = deque(maxlen=max_ep_num)
        self.max_ep_len = 300
        self.ep_len_dist = np.zeros((self.max_ep_len,), np.float32)
        self.bptt = T

        self.env = env
        self._action_space = env.action_space
        self._ob_space = env.observation_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_ep_num,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_paths(self, paths):
        for path in paths:
            path_len = len(path['actions'])
            if path_len > self.bptt:
                if len(self.ep_buffer) == self.max_ep_num:
                    ep_del_len = self.ep_buffer[0][1]
                    self.ep_len_dist[ep_del_len] -= ep_del_len
                self.ep_buffer.append((path, path_len))
                self.ep_len_dist[path_len] += path_len
                self._advance()

    def random_batch(self, batch_size):
        # random ep_len(total), random t with bptt = 1
        ep_len = np.random.choice(a=self.max_ep_len, size=None, p=self.ep_len_dist / np.sum(self.ep_len_dist))
        ep_tmp = []
        for ep in self.ep_buffer:
            if ep[1] == ep_len:
                ep_tmp.append(ep[0])

        if batch_size < len(ep_tmp):
            sampled_ep = random.sample(ep_tmp, batch_size)
        else:
            sampled_ep = ep_tmp

        observations = np.zeros((batch_size, 4+2+1+self.bptt*(4+3)), np.float32)
        actions = np.zeros((batch_size, 3), np.float32)
        rewards = np.zeros((batch_size, 1), np.float32)
        terminals = np.zeros((batch_size, 1), np.float32)
        next_observations = np.zeros((batch_size, 4+2+1+self.bptt*(4+3)), np.float32)
        for i in range(len(sampled_ep)):
            ep = sampled_ep[i]
            o = ep['observations']
            a = ep['actions']
            r = ep['rewards']
            next_o = ep['next_observations']
            term = ep['terminals']
            t = np.random.choice(a=ep_len-self.bptt, size=None) + self.bptt
            observations[i, 0:4] = o[t][0:4]
            observations[i, 4:6] = o[t][4:6]
            observations[i, 6:7] = o[t][6:7]
            for k, j in enumerate(range(t-self.bptt, t)):
                observations[i, 7 + 7 * k:11 + 7 * k] = o[j][0:4]
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[j]] = 1
                observations[i, 11 + 7 * k:14 + 7 * k] = a_onehot
            a_onehot = np.zeros((3,), np.float32)
            a_onehot[a[t]] = 1
            actions[i] = a_onehot
            rewards[i] = r[t]
            terminals[i] = term[t]
            next_observations[i, 0:4] = next_o[t][0:4]
            next_observations[i, 4:6] = next_o[t][4:6]
            next_observations[i, 6:7] = next_o[t][6:7]
            for k, j in enumerate(range(t-self.bptt+1, t+1)):
                next_observations[i, 7 + 7 * k:11 + 7 * k] = o[j][0:4]
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[j]] = 1
                next_observations[i, 11 + 7 * k:14 + 7 * k] = a_onehot

        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_observations=next_observations,
        )
        return batch


class EnvReplayBufferM0(SimpleReplayBuffer):
    def __init__(
            self,
            max_ep_num,
            env,
            env_info_sizes=None
    ):
        self.max_ep_num = max_ep_num
        from collections import deque
        self.ep_buffer = deque(maxlen=max_ep_num)
        self.max_ep_len = 300
        self.ep_len_dist = np.zeros((self.max_ep_len,), np.float32)
        self.bptt = 20
        self.min_bptt = 8
        self.option = 3

        self.env = env
        self._action_space = env.action_space
        self._ob_space = env.observation_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=1,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_paths(self, paths):
        for path in paths:
            if len(self.ep_buffer) == self.max_ep_num:
                ep_del_len = self.ep_buffer[0][1]
                self.ep_len_dist[ep_del_len] -= ep_del_len
            path_len = len(path['actions'])
            self.ep_buffer.append((path, path_len))
            self.ep_len_dist[path_len] += path_len

    def random_batch(self, batch_size):
        if self.option == 1:  # random ep, random t
            observations = np.zeros((batch_size, 4 + 2 + 1), np.float32)
            actions = np.zeros((batch_size, 3), np.float32)
            rewards = np.zeros((batch_size, 1), np.float32)
            terminals = np.zeros((batch_size, 1), np.float32)
            next_observations = np.zeros((batch_size, 4 + 2 + 1), np.float32)
            sampled_ep = random.sample(self.ep_buffer, batch_size)
            print(list(map(list, zip(*sampled_ep)))[1])
            for i in range(len(sampled_ep)):
                ep, ep_len = sampled_ep[i]
                o = ep['observations']
                a = ep['actions']
                r = ep['rewards']
                next_o = ep['next_observations']
                term = ep['terminals']
                t = np.random.choice(a=ep_len, size=None)
                observations[i, 0:4] = o[t][0:4]
                observations[i, 4:6] = o[t][4:6]
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[t]] = 1
                actions[i] = a_onehot
                rewards[i] = r[t]
                terminals[i] = term[t]
                next_observations[i, 0:4] = next_o[t][0:4]
                next_observations[i, 4:6] = next_o[t][4:6]
        elif self.option == 2:  # random ep_len(total), random t with bptt = 1
            self.ep_len_dist[1] = 0
            ep_len = np.random.choice(a=self.max_ep_len, size=None, p=self.ep_len_dist / np.sum(self.ep_len_dist))
            ep_tmp = []
            for ep in self.ep_buffer:
                if ep[1] == ep_len:
                    ep_tmp.append(ep[0])

            if batch_size < len(ep_tmp):
                sampled_ep = random.sample(ep_tmp, batch_size)
            else:
                sampled_ep = ep_tmp

            bptt = 1
            observations = np.zeros((batch_size, 4+2+1+bptt*(4+3)), np.float32)
            actions = np.zeros((batch_size, 3), np.float32)
            rewards = np.zeros((batch_size, 1), np.float32)
            terminals = np.zeros((batch_size, 1), np.float32)
            next_observations = np.zeros((batch_size, 4+2+1+bptt*(4+3)), np.float32)
            for i in range(len(sampled_ep)):
                ep = sampled_ep[i]
                o = ep['observations']
                a = ep['actions']
                r = ep['rewards']
                next_o = ep['next_observations']
                term = ep['terminals']
                t = np.random.choice(a=ep_len-bptt, size=None) + bptt
                observations[i, 0:4] = o[t][0:4]
                observations[i, 4:6] = o[t][4:6]
                observations[i, 6:7] = o[t][6:7]
                for k, j in enumerate(range(t-bptt, t)):
                    observations[i, 7 + 7 * k:11 + 7 * k] = o[j][0:4]
                    a_onehot = np.zeros((3,), np.float32)
                    a_onehot[a[j]] = 1
                    observations[i, 11 + 7 * k:14 + 7 * k] = a_onehot
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[t]] = 1
                actions[i] = a_onehot
                rewards[i] = r[t]
                terminals[i] = term[t]
                next_observations[i, 0:4] = next_o[t][0:4]
                next_observations[i, 4:6] = next_o[t][4:6]
                next_observations[i, 6:7] = next_o[t][6:7]
                for k, j in enumerate(range(t-bptt+1, t+1)):
                    next_observations[i, 7 + 7 * k:11 + 7 * k] = o[j][0:4]
                    a_onehot = np.zeros((3,), np.float32)
                    a_onehot[a[j]] = 1
                    next_observations[i, 11 + 7 * k:14 + 7 * k] = a_onehot
        elif self.option == 3:  # random ep_len(total), random t with bptt dependent
            ep_len = np.random.choice(a=self.max_ep_len, size=None, p=self.ep_len_dist / np.sum(self.ep_len_dist))
            ep_tmp = []
            for ep in self.ep_buffer:
                if ep[1] == ep_len:
                    ep_tmp.append(ep[0])

            if ep_len > self.bptt + 1:
                batch_size = 2 * min(len(ep_tmp), batch_size)
                sampled_ep_1 = random.sample(ep_tmp, batch_size // 2)
                sampled_ep_2 = random.sample(ep_tmp, batch_size // 2)
                sampled_ep = sampled_ep_1 + sampled_ep_2
                sampled_t_1 = np.random.choice(a=ep_len - self.bptt - 1, size=batch_size) + self.bptt
                sampled_t_2 = np.random.choice(a=1, size=batch_size) + ep_len - 1
                sampled_t = np.concatenate((sampled_t_1, sampled_t_2))
                bptt = self.bptt
            elif ep_len > self.min_bptt:
                batch_size = min(len(ep_tmp), batch_size)
                sampled_ep = random.sample(ep_tmp, batch_size)
                sampled_t = np.random.choice(a=ep_len - self.min_bptt, size=batch_size) + self.min_bptt
                bptt = self.min_bptt
            else:
                batch_size = min(len(ep_tmp), batch_size)
                sampled_ep = random.sample(ep_tmp, batch_size)
                sampled_t = np.random.choice(a=ep_len, size=batch_size)
                bptt = min(self.bptt, np.min(sampled_t))

            observations = np.zeros((batch_size, 4 + 2 + 1 + bptt * (4 + 3)), np.float32)
            actions = np.zeros((batch_size, 3), np.float32)
            rewards = np.zeros((batch_size, 1), np.float32)
            terminals = np.zeros((batch_size, 1), np.float32)
            next_observations = np.zeros((batch_size, 4 + 2 + 1 + bptt * (4 + 3)), np.float32)
            for i in range(len(sampled_ep)):
                ep = sampled_ep[i]
                t = sampled_t[i]
                o = ep['observations']
                a = ep['actions']
                r = ep['rewards']
                next_o = ep['next_observations']
                term = ep['terminals']

                # indexing o and next_o is different since o has different
                # length at first iteration
                observations[i, 0:4] = o[t][0:4]
                observations[i, 4:6] = o[t][4:6]
                observations[i, 6:7] = o[t - bptt][6:7]
                # observations[i, 6:7] = 0
                for k, j in enumerate(range(t - bptt, t)):
                    observations[i, 7 + 7 * k:11 + 7 * k] = o[j][0:4]
                    a_onehot = np.zeros((3,), np.float32)
                    a_onehot[a[j]] = 1
                    observations[i, 11 + 7 * k:14 + 7 * k] = a_onehot
                a_onehot = np.zeros((3,), np.float32)
                a_onehot[a[t]] = 1
                actions[i] = a_onehot
                rewards[i] = r[t]
                terminals[i] = term[t]
                next_observations[i, 0:4] = next_o[t][0:4]
                next_observations[i, 4:6] = next_o[t][4:6]
                next_observations[i, 6:7] = next_o[t - bptt][6:7]
                # next_observations[i, 6:7] = 0
                for k, j in enumerate(range(t - bptt + 1, t + 1)):
                    next_observations[i, 7 + 7 * k:11 + 7 * k] = o[j][0:4]
                    a_onehot = np.zeros((3,), np.float32)
                    a_onehot[a[j]] = 1
                    next_observations[i, 11 + 7 * k:14 + 7 * k] = a_onehot


        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_observations=next_observations,
        )
        return batch


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )


class EnvReplayBufferEpforV(SimpleReplayBuffer):
    def __init__(
            self,
            max_ep_num,
            env,
            env_info_sizes=None
    ):
        self.max_ep_num = max_ep_num
        from collections import deque
        self.ep_buffer = deque(maxlen=max_ep_num)
        self.max_ep_len = 300

        self.env = env
        self._action_space = env.action_space
        self._ob_space = env.observation_space
        self.t = 1000
        self.max_sampled_ep_len = 0

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=1,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )
        self._max_replay_buffer_size = max_ep_num

    def add_paths(self, paths):
        for path in paths:
            path_len = len(path['actions'])
            self.ep_buffer.append((path, path_len))
            self._advance()

    def random_batch(self, batch_size):
        if self.t >= self.max_sampled_ep_len:
            self.sampled_ind = np.random.choice(a=self._size, size=batch_size)
            self.sampled_ep = []
            self.sampled_ep_len = []
            for ind in self.sampled_ind:
                self.sampled_ep.append(self.ep_buffer[ind][0])
                self.sampled_ep_len.append(self.ep_buffer[ind][1])
            self.sampled_ep_len = np.array(self.sampled_ep_len)
            self.t = 0
            self.max_sampled_ep_len = np.max(self.sampled_ep_len)

        batch_ep_index = np.argwhere(self.sampled_ep_len > self.t).flatten()
        batch_ep = [self.sampled_ep[ind] for ind in batch_ep_index]
        new_batch_size = len(batch_ep_index)
        observations = np.zeros((new_batch_size, self._observation_dim), np.float32)
        actions = np.zeros((new_batch_size, 3), np.float32)
        rewards = np.zeros((new_batch_size, 1), np.float32)
        terminals = np.zeros((new_batch_size, 1), np.float32)
        next_observations = np.zeros((new_batch_size, self._observation_dim), np.float32)

        for i in range(new_batch_size):
            ep = batch_ep[i]
            observations[i] = ep['observations'][self.t]
            a_onehot = np.zeros((3,), np.float32)
            a_onehot[ep['actions'][self.t]] = 1
            actions[i] = a_onehot
            rewards[i] = ep['rewards'][self.t]
            terminals[i] = ep['terminals'][self.t]
            next_observations[i] = ep['next_observations'][self.t]

        batch = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_observations=next_observations,
        )
        self.t += 1
        return batch