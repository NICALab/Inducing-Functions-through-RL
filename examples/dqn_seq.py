import argparse
import gym
from torch import nn as nn
import numpy as np

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicyM0
from rlkit.torch.dqn.dqn import DQNTrainerM0
from rlkit.torch.networks.custom import BaselineSeq
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBufferSeq, EnvReplayBufferSeq_, EnvReplayBufferEp
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollectorM0
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from env_survive.env_survive_tmp import EnvSurvive, T, pred_status

raw_vision = True
true_hidden = False
different_lr = True


def experiment(variant):
    expl_env = EnvSurvive(path='./env_survive/mnist', seed=0,
                          raw_vision=raw_vision,
                          memory_task=True,
                          true_hidden=true_hidden)
    eval_env = EnvSurvive(path='./env_survive/mnist', seed=0,
                          raw_vision=raw_vision, memory_task=True, true_hidden=true_hidden)

    qf = BaselineSeq(raw_vision=raw_vision, pred_status=pred_status)
    target_qf = BaselineSeq(raw_vision=raw_vision, pred_status=pred_status)

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
        different_lr=different_lr,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBufferEp(
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
    num_epochs = 2000
    variant = dict(
        algorithm="DQN",
        version="normal",
        replay_buffer_size=int(1E5),
        algorithm_kwargs=dict(
            num_epochs=num_epochs,
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
        survive_kwargs=dict(
            raw_vision=raw_vision,
            true_hidden=true_hidden,
            T=T,
            pred_status=pred_status,
            different_lr=different_lr,
        )
    )
    if raw_vision:
        exp_prefix = 'task_b'
    else:
        exp_prefix = 'task_a'
    exp_id = 2

    setup_logger(exp_prefix=exp_prefix,
                 variant=variant,
                 text_log_file="debug.log",
                 variant_log_file="variant.json",
                 tabular_log_file="progress.csv",
                 snapshot_mode="gap_and_last",
                 snapshot_gap=num_epochs / 10,
                 log_tabular_only=False,
                 log_dir=None,
                 git_infos=None,
                 script_name=None,
                 # **create_log_dir_kwargs
                 base_log_dir='./data',
                 exp_id=exp_id,
                 seed=T)  # BPTT
    ptu.set_gpu_mode(True)
    experiment(variant)


