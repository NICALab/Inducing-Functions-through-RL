import argparse
import gym
from torch import nn as nn
import numpy as np

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicyM0
from rlkit.torch.dqn.dqn import DQNTrainerM0
from rlkit.torch.networks.custom import BaselineM0
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBufferM0
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollectorM0
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from env_survive.env_survive import EnvSurvive, vision_map, action_map


def experiment(variant):
    expl_env = EnvSurvive(path='./env_survive/mnist', seed=0, raw_vision=False, memory_task=True)
    eval_env = EnvSurvive(path='./env_survive/mnist', seed=0, raw_vision=False, memory_task=True)

    qf = BaselineM0()
    target_qf = BaselineM0()

    '''
    # M0 with known mlp
    # load Baseline-V0
    import torch
    f_path = './data/dqn-survive/dqn-survive_2021_01_12_17_23_14_0000--s-0/itr_1800.pkl'
    data = torch.load(f_path)
    BaselineV0 = data['evaluation/policy'].qf.mlp
    source = BaselineV0
    for target_param, param in zip(qf.mlp.parameters(), source.parameters()):
        target_param.data.copy_(param)
    for target_param, param in zip(target_qf.mlp.parameters(), source.parameters()):
        target_param.data.copy_(param)
    '''
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicyM0(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
    )
    eval_path_collector = MdpPathCollectorM0(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollectorM0(
        expl_env,
        expl_policy,
    )
    trainer = DQNTrainerM0(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBufferM0(
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
    variant = dict(
        algorithm="DQN",
        version="normal",
        replay_buffer_size=int(1E5),
        algorithm_kwargs=dict(
            num_epochs=2000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=2000,
            max_path_length=500,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        ),
    )
    setup_logger(exp_prefix='dqn-survive',
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
                 exp_id=1,
                 seed=0)  # baseline m0
    ptu.set_gpu_mode(True)
    experiment(variant)


'''
for np1, np2 in zip(target.named_parameters(), source.named_parameters()):
    n1 = np1[0]  # parameter name
    p1 = np1[1]  # paramter value
    n2 = np2[0]
    print(n1, n2)
'''