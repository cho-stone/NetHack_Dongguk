"""
import os
import sys
import time
from collections import deque

import gym
import numpy as np
import scipy
import torch

import rl_baseline.tasks_nle
import rl_baseline.encoders_nle
import rl_baseline.env_nle

from rlaif.reward_model import create_reward_model
from sample_factory.algorithms.appo.actor_worker import (
    transform_dict_observations)
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import (arg_parser, parse_args,
    load_from_checkpoint)
from sample_factory.algorithms.utils.multi_agent_wrapper import (
    MultiAgentWrapper)
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict, str2bool
"""

import torch
import numpy as np
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic

class MultiTaskAgent :

    def __init__(self, cfg, env) :
        self.models = np.empty(6, dtype=object)
        self.env = env
        self.cfg = cfg
        self.get_models()

    def load_model(self, cfg, i) :
        actor_critic = create_actor_critic(
            cfg, 
            self.env.observation_space, 
            self.env.action_space
        )
        device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
        actor_critic.model_to_device(device)

        checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, i))
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])
        return actor_critic

    #load multi_task_model
    def get_models(self) :
        """
        policy_index
        0 : score
        1 : gold
        2 : staircase
        3 : eat
        4 : scout
        5 : pet
        """
        for i in range(6) :
            self.models[i] = self.load_model(self.cfg, i)
            print(i, " model load")
    
    def forward_models(self, obs_torch, rnn_states) :
        policy_outputs = np.empty(6, dtype=object)
        for i in range(6) :
            policy_outputs[i] = self.models[i](obs_torch, rnn_states)
        return policy_outputs

    #load calibrating mlp

    #calculate calibrated reward

    #task selector
    def select_task(self, policy_outputs) :
        values = np.zeros(6)
        for i in range(6) :
            values[i] = policy_outputs[i].values.item()
        print(values, np.argmax(values))
        actions = policy_outputs[np.argmax(values)].actions
        rnn_states = policy_outputs[np.argmax(values)].rnn_states
        return actions, rnn_states

    def get_outputs(self, obs_torch, rnn_states) :
        return self.select_task(self.forward_models(obs_torch, rnn_states))
"""
def add_extra_params(parser):
    p = parser
    p.add_argument("--sleep", default=0.0, type=float, 
                   help="Controls the speed at which rendering happens.")
    p.add_argument("--render", default=True, type=str2bool, 
                   help="To render or not the game.")
def parse_all_args(argv=None, evaluation=True):
    parser = arg_parser(argv, evaluation=evaluation)
    add_extra_params(parser)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg

cfg = parse_all_args()

cfg = load_from_checkpoint(cfg)
cfg.env_frameskip = 1
cfg.num_envs = 1
env_config = AttrDict({'worker_index': 0, 'vector_index': 0})
env = create_env(cfg.env, cfg=cfg, env_config=env_config)
env = MultiAgentWrapper(env)

a = MultiTaskAgent(cfg, env)

device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
obs = env.reset()
rnn_states = torch.zeros(
    [env.num_agents, get_hidden_size(cfg)], 
    dtype=torch.float32, 
    device=device
)

obs_torch = AttrDict(transform_dict_observations(obs))
for key, x in obs_torch.items():
    obs_torch[key] = torch.from_numpy(x).to(device).float()
policy_outputs = a.forward_models(obs_torch, rnn_states)

#print(a.models)
for i in range(6) :
    print(a.select_task(policy_outputs))
"""