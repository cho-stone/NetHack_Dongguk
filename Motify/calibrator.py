import os
import sys
import time
from collections import deque

import gym
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

import rl_baseline.tasks_nle
import rl_baseline.encoders_nle
import rl_baseline.env_nle

from sample_factory.algorithms.appo.actor_worker import (transform_dict_observations)
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import (arg_parser, parse_args,load_from_checkpoint)
from sample_factory.algorithms.utils.multi_agent_wrapper import (MultiAgentWrapper)
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict, str2bool
from sample_factory.algorithms.appo.model_utils import create_encoder, normalize_obs
from sample_factory.utils.timing import Timing

class Calibrator(nn.Module):
    def __init__(self, make_encoder, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = make_encoder()   
        self.mlp = nn.Sequential(
            nn.Linear(795, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 23)
            )
    def forward_head(self, obs_dict, normalize=True):
        if normalize:
            normalize_obs(obs_dict, self.cfg)
        x = self.encoder(obs_dict)
        return x
    def forward(self, obs_dict):
        x = self.forward_head(obs_dict)
        x = self.mlp(x)
        return x

def create_calibrator(cfg, obs_space, timing = None):
    if timing is None:
        timing = Timing()

    def make_encoder():
        return create_encoder(cfg, obs_space, timing)
    return Calibrator(make_encoder, cfg)

def train(cfg) :
    cfg = load_from_checkpoint(cfg)

    cfg.env_frameskip = 1
    cfg.num_envs = 1

    env_config = AttrDict({'worker_index': 0, 'vector_index': 0})
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = MultiAgentWrapper(env)

    actor_critic = create_actor_critic(
        cfg, 
        env.observation_space, 
        env.action_space
    )
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, cfg.policy_index))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    calibrator = create_calibrator(cfg, env.observation_space)
    calibrator.to(device)

    max_score = 100
    finished_episode = [False] * env.num_agents
    episode_num_frames = 0
    num_frames = 0
    num_episodes = 0
    agent_i = 0

    obs = env.reset()
    rnn_states = torch.zeros(
        [env.num_agents, get_hidden_size(cfg)], 
        dtype=torch.float32, 
        device=device
    )

    calibrator_dir = "train_dir/calibrator_dir"
    os.makedirs(calibrator_dir, exist_ok=True)
    i=0
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=0.001)
    #score = 0

    max_num_frames = 1e6
    while num_frames < max_num_frames:
        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(obs))
            print("frame : ", num_frames)

            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            policy_outputs = actor_critic(obs_torch, rnn_states)
            actions = policy_outputs.actions
            action_distribution = policy_outputs.action_distribution.probs[0][actions]
            actions = actions.cpu().numpy()

            obs, rew, done, infos = env.step(actions)

        new_score = infos[0].get('score', None)
        if episode_num_frames > 0:
            score_diff = new_score - score
            score_diff = 100 if score_diff > 100 else score_diff if score_diff > 0 else 0  
            output = torch.zeros(23)
            output[actions] = score_diff / max_score / action_distribution
            output = output.unsqueeze(0).to(device)
            #input: obs_torch, output: output
            #calibrator train
            hypothesis = calibrator(obs_torch)
            cost = F.cross_entropy(input=hypothesis, target=output)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        score = new_score

        
        num_frames += 1
        episode_num_frames += 1

        if done[agent_i]:
            finished_episode[agent_i] = True
            episode_num_frames = 0
            num_episodes += 1
            rnn_states[agent_i] = torch.zeros(
                [get_hidden_size(cfg)], 
                dtype=torch.float32, 
                device=device
            )
        
        if (num_frames%100 == 0):
            i+=1
            i = 0 if i>4 else i

            save_path = os.path.join(calibrator_dir, f'checkpoint_p{i}.pth')
            torch.save(calibrator.state_dict(), save_path)
            print("sdfasdf")
    env.close()

    return ExperimentStatus.SUCCESS

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


def main():
    cfg = parse_all_args()
    _ = train(cfg)
    return


if __name__ == '__main__':
    sys.exit(main())