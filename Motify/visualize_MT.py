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

from scripts.MultiTaskAgent import MultiTaskAgent


def enjoy(cfg, max_num_frames=1e6, target_num_episodes=100):
    """
    This is a modified version of original appo.enjoy_appo.enjoy function.
    """

    cfg = load_from_checkpoint(cfg)

    cfg.env_frameskip = 1
    cfg.num_envs = 1

    env_config = AttrDict({'worker_index': 0, 'vector_index': 0})
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = MultiAgentWrapper(env)

    mt_model = MultiTaskAgent(cfg, env)
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')

    rew_checkpoints = LearnerWorker.get_checkpoints(os.path.join(cfg.reward_dir,
                                                            f'checkpoint_p0'))

    if len(rew_checkpoints) > 0:
        checkpoint_dict = LearnerWorker.load_checkpoint(
            rew_checkpoints, 
            device, 
            checkpoint_num=cfg.checkpoint_num
        )

        bias_weight_name = 'action_parameterization.distribution_linear.bias'
        action_space = checkpoint_dict['model'][bias_weight_name].shape[0]
        action_space = gym.spaces.Discrete(action_space)

        reward_model = create_reward_model(
            cfg, 
            env.observation_space, 
            action_space
        )
        reward_model.model_to_device(device)
        reward_model.load_state_dict(checkpoint_dict['model'])

        mean = checkpoint_dict['reward_mean'].item()
        var = checkpoint_dict['reward_var'].item()
        log.info("Reward function loaded...\n")
    else:
        reward_model = create_reward_model(
            cfg, 
            env.observation_space, 
            env.action_space
        )
        reward_model.model_to_device(device)

        mean = 0
        var = 1
        log.info("No reward function loaded...\n")

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    episode_reward = np.zeros(env.num_agents)
    finished_episode = [False] * env.num_agents

    episode_msgs = {}
    episode_num_frames = 0
    num_frames = 0
    num_episodes = 0
    agent_i = 0

    obs = env.reset()
    k = env.get_seeds()
    rnn_states = np.empty(5, dtype=object)
    for i in range(5):
        rnn_states[i] = torch.zeros(
            [env.num_agents, get_hidden_size(cfg)], 
            dtype=torch.float32, 
            device=device
        )

    while num_frames < max_num_frames and num_episodes < target_num_episodes:
        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(obs))


            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            actions, rnn_states = mt_model.get_outputs(obs_torch, rnn_states)
            actions = actions.cpu().numpy()

            obs, rew, done, infos = env.step(actions)

            reward_obs = obs[0].copy()
            
            int_reward = reward_model(
                reward_obs, 
                add_dim=True
            ).rewards.cpu().numpy()[0][0]
            msg = bytes(obs[0]['message'])
            norm_int_reward = (int_reward - mean) / (var)**(1/2)

            if msg not in episode_msgs:
                episode_msgs[msg] = (1, int_reward, norm_int_reward)
            else:
                count, int_r, norm_int_r = episode_msgs[msg]
                episode_msgs[msg] = (count+1, int_r, norm_int_r) 


            episode_reward += rew
            num_frames += 1
            episode_num_frames += 1

            if done[agent_i]:
                finished_episode[agent_i] = True
                episode_rewards[agent_i].append(episode_reward[agent_i])

                print("\n" * 27)
                print("===============Top messages=================")
                sorted_msgs = sorted(
                    episode_msgs.items(), 
                    key=lambda x: x[1][0], 
                    reverse=True
                )[:100]
                for msg, value in sorted_msgs:
                    print(f' {msg.decode()} {value[2]:.3f} ')
                print("===============Top messages=================")
                print(f"Episode finished at {num_frames} frames. " 
                        f"Return: {episode_reward[agent_i]:.3f}")

                episode_msgs = {}
                episode_num_frames = 0
                episode_reward[agent_i] = 0
                num_episodes += 1
                rnn_states = np.empty(6, dtype=object)
                for i in range(6):
                    rnn_states[i] = torch.zeros(
                        [env.num_agents, get_hidden_size(cfg)], 
                        dtype=torch.float32, 
                        device=device
                    )
                input("Press 'Enter' to continue...")

            if cfg.render:
                print(f"Timestep: {num_frames} Reward: {rew[0]:.3f} "
                    f"Return: {episode_reward[0]:.3f} "
                    f"Intrinsic reward: {int_reward:.3f} "
                    f"Norm Intrinsic reward: {norm_int_reward:.3f}")

                env.render()

                print("\033[%dA" % 27)

                time.sleep(cfg.sleep)

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)

def add_extra_params(parser):
    """
    RL 에이전트를 시각화하기 위한 추가 명령행 인자를 지정
    """
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
    """평가를 위한 진입점"""
    cfg = parse_all_args()
    _ = enjoy(cfg)
    return


if __name__ == '__main__':
    sys.exit(main())
