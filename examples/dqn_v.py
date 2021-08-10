import argparse
import gym
from torch import nn as nn
import numpy as np

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks.custom import BaselineV
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer, EnvReplayBufferEpforV
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from env_survive.env_survive import EnvSurvive, vision_map, action_map

raw_vision = True
raw_status = False
different_sampling = False

def experiment(variant):
    expl_env = EnvSurvive(path='./env_survive/mnist', seed=0, raw_vision=raw_vision)
    eval_env = EnvSurvive(path='./env_survive/mnist', seed=0, raw_vision=raw_vision)

    qf = BaselineV(raw_vision=raw_vision, raw_status=raw_status)
    target_qf = BaselineV(raw_vision=raw_vision, raw_status=raw_status)
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    if different_sampling:
        replay_buffer = EnvReplayBufferEpforV(
            variant['replay_buffer_size'],
            expl_env,
        )
    else:
        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_size'],
            expl_env,
        )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    if different_sampling:
        buffer_size = int(1E5)
    else:
        buffer_size = int(1E6)
    variant = dict(
        algorithm="DQN",
        version="normal",
        replay_buffer_size=buffer_size,
        algorithm_kwargs=dict(
            num_epochs=4000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=500,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        ),
    )
    if raw_vision:
        if raw_status:
            exp_prefix = 'task_b'
        else:
            exp_prefix = 'task_b_vision_only'
    else:
        exp_prefix = 'task_a'
    if raw_status:
        exp_id = 0
    else:
        exp_id = 1

    setup_logger(exp_prefix=exp_prefix,
                 variant=variant,
                 text_log_file="debug.log",
                 variant_log_file="variant.json",
                 tabular_log_file="progress.csv",
                 snapshot_mode="gap_and_last",
                 snapshot_gap=200,
                 log_tabular_only=False,
                 log_dir=None,
                 git_infos=None,
                 script_name=None,
                 # **create_log_dir_kwargs
                 base_log_dir='./data',
                 exp_id=exp_id,
                 seed=0)
    ptu.set_gpu_mode(True)
    experiment(variant)
