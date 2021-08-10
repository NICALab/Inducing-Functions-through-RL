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
from rlkit.data_management.env_replay_buffer import EnvReplayBufferSequence, EnvReplayBufferRandom
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollectorM0
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from env_survive.env_survive_real import EnvSurvive, T, pred_status, load_data

# Notice that if sequential_update is True, T is meaningless
# however, fixed_sequence becomes important
# For sequential_update is False, which is random sampling, T is important
raw_vision = True
true_hidden = False
different_lr = False
sequential_update = True
fixed_sequence = 60
fixed_sequence_use_prediction = True
memory_variation = 1  # 0 if mlp_tmp, 1 if mlp_ext, 2 if (mlp_h, mlp_s)
sub_path = ''

def experiment(variant):
    # load data in advance because of memory problem
    path = './env_survive/mnist' + sub_path
    dataset, vision_shape = load_data(path)
    expl_env = EnvSurvive(dataset=dataset, vision_shape=vision_shape, seed=0,
                          raw_vision=raw_vision, sequential_update=sequential_update, true_hidden=true_hidden)
    eval_env = EnvSurvive(dataset=dataset, vision_shape=vision_shape, seed=0,
                          raw_vision=raw_vision, sequential_update=sequential_update, true_hidden=true_hidden)

    qf = BaselineSeq(raw_vision=raw_vision, pred_status=pred_status, memory_variation=memory_variation)
    target_qf = BaselineSeq(raw_vision=raw_vision, pred_status=pred_status, memory_variation=memory_variation)

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
    if sequential_update:
        replay_buffer = EnvReplayBufferSequence(
            variant['replay_buffer_size'],
            expl_env,
            fixed_sequence,
            fixed_sequence_use_prediction,
        )
    else:
        replay_buffer = EnvReplayBufferRandom(
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
    num_epochs = 3000
    if sequential_update:
        replay_buffer_size = int(1E5)
    else:
        replay_buffer_size = int(1E6)

    variant = dict(
        algorithm="DQN",
        version="normal",
        replay_buffer_size=replay_buffer_size,
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
            pred_status=pred_status,
            different_lr=different_lr,
            sequential_update=sequential_update,
            T=T,
            fixed_sequence=fixed_sequence,
            fixed_sequence_use_prediction=fixed_sequence_use_prediction,
            memory_variation=memory_variation,
            sub_path=sub_path,
        )
    )

    if raw_vision:
        if sequential_update:
            if memory_variation == 0:
                exp_prefix = 'task_b_sequence'
            elif memory_variation == 1:
                exp_prefix = 'task_b_sequence_ext'
                if fixed_sequence_use_prediction:
                    exp_prefix = exp_prefix + '_use_pred'
            elif memory_variation == 2:
                exp_prefix = 'task_b_sequence_sep'
            seed = fixed_sequence
        else:
            if memory_variation == 0:
                exp_prefix = 'task_b_random'
            elif memory_variation == 1:
                exp_prefix = 'task_b_random_ext'
            elif memory_variation == 2:
                exp_prefix = 'task_b_random_sep'
            seed = T
    else:
        exp_prefix = 'task_a_real'
    exp_id = 2

    setup_logger(exp_prefix=exp_prefix,
                 variant=variant,
                 text_log_file="debug.log",
                 variant_log_file="variant.json",
                 tabular_log_file="progress.csv",
                 snapshot_mode="gap_and_last",
                 snapshot_gap=500,
                 log_tabular_only=False,
                 log_dir=None,
                 git_infos=None,
                 script_name=None,
                 # **create_log_dir_kwargs
                 base_log_dir='./data',
                 exp_id=exp_id,
                 seed=seed)  # BPTT
    ptu.set_gpu_mode(True)
    experiment(variant)


