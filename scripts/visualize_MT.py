import os #운영 체제와 상호작용하기 위한 모듈
import sys #파이썬 인터프리터와 상호작용하기 위한 모듈
import time #시간 관련 기능을 제공하는 모듈
from collections import deque #효율적인 큐(Queue) 자료구조를 제공하는 모듈로, 양쪽에서 삽입 및 삭제가 빠르다

import gym #강화 학습 환경을 만들기 위한 라이브러리
import numpy as np #수치 계산을 위한 라이브러리로, 다차원 배열을 처리
import scipy #과학 및 공학 계산을 위한 라이브러리
import torch #딥러닝을 위한 라이브러리로, 텐서 연산 및 신경망 구축을 지원

#모델과 환경을 등록하기 위해 필요한 모듈
import rl_baseline.tasks_nle
import rl_baseline.encoders_nle
import rl_baseline.env_nle

from rlaif.reward_model import create_reward_model #보상 모델을 생성하기 위한 함수
from sample_factory.algorithms.appo.actor_worker import (
    transform_dict_observations) #관찰값을 변환하는 함수로, 액터-크리틱 알고리즘에서 사용
from sample_factory.algorithms.appo.learner import LearnerWorker #학습 작업을 처리하는 클래스
from sample_factory.algorithms.appo.model import create_actor_critic #액터-크리틱 모델을 생성하는 함수
from sample_factory.algorithms.appo.model_utils import get_hidden_size #모델의 히든 레이어 크기를 반환하는 유틸리티 함수
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus #실험 상태를 관리하는 클래스
from sample_factory.algorithms.utils.arguments import (arg_parser, parse_args,
    load_from_checkpoint)
from sample_factory.algorithms.utils.multi_agent_wrapper import (
    MultiAgentWrapper) #여러 에이전트를 관리하기 위한 래퍼 클래스
from sample_factory.envs.create_env import create_env #환경을 생성하기 위한 함수
from sample_factory.utils.utils import log, AttrDict, str2bool

from scripts.MultiTaskAgent import MultiTaskAgent


def enjoy(cfg, max_num_frames=1e6, target_num_episodes=100): #설정, 최대 프레임 수, 목표 에피소드 수를 받는다
    """
    This is a modified version of original appo.enjoy_appo.enjoy function.
    """

    cfg = load_from_checkpoint(cfg) #체크포인트에서 설정(cfg)을 불러온다.

    cfg.env_frameskip = 1
    cfg.num_envs = 1

    #주어진 설정에 따라 환경을 생성헌더,
    env_config = AttrDict({'worker_index': 0, 'vector_index': 0})
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)

    #샘플 팩토리는 멀티 에이전트 환경을 기본으로 하지만,
    #단일 에이전트 환경을 멀티 에이전트 래퍼로 감쌀 수 있다
    env = MultiAgentWrapper(env)

    mt_model = MultiTaskAgent(cfg, env)
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')

    #보상 모델을 로드하고, 필요시 보상 체크포인트를 불러온다.
    rew_checkpoints = LearnerWorker.get_checkpoints(os.path.join(cfg.reward_dir,
                                                            f'checkpoint_p0'))

    if len(rew_checkpoints) > 0: #보상 체크포인트가 존재하는지 확인한다
        checkpoint_dict = LearnerWorker.load_checkpoint(
            rew_checkpoints, 
            device, 
            checkpoint_num=cfg.checkpoint_num
        ) #보상 체크포인트에서 상태를 로드한다

        #행동 공간이 올바른지 확인한다. 보상 함수는 행동 정보를 사용하지 않지만, 행동 공간에 의존한다.
        #모델에서 행동 공간의 크기를 확인하고, 이를 사용해 gym의 이산 공간으로 설정한다.
        bias_weight_name = 'action_parameterization.distribution_linear.bias'
        action_space = checkpoint_dict['model'][bias_weight_name].shape[0]
        action_space = gym.spaces.Discrete(action_space)

        #주어진 설정과 환경 관찰 공간, 행동 공간을 사용해 보상 모델을 생성한다.
        reward_model = create_reward_model(
            cfg, 
            env.observation_space, 
            action_space
        )
        reward_model.model_to_device(device)
        reward_model.load_state_dict(checkpoint_dict['model']) #로드한 체크포인트의 모델 상태를 보상 모델에 적용한다.

        #보상 함수의 평균과 분산을 불러온다.
        mean = checkpoint_dict['reward_mean'].item()
        var = checkpoint_dict['reward_var'].item()
        log.info("Reward function loaded...\n")
    else: #보상 체크포인트가 존재하지 않을 경우
        reward_model = create_reward_model(
            cfg, 
            env.observation_space, 
            env.action_space
        ) #새로운 보상 모델을 생성한다.
        reward_model.model_to_device(device)

        mean = 0
        var = 1
        log.info("No reward function loaded...\n")

    #각 에이전트에 대해 최대 100개의 보상을 저장할 수 있는 deque 리스트를 생성하고, 에피소드 보상과 에피소드 완료 여부 리스트를 초기화한다.
    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    episode_reward = np.zeros(env.num_agents)
    finished_episode = [False] * env.num_agents

    episode_msgs = {}
    episode_num_frames = 0
    num_frames = 0
    num_episodes = 0
    agent_i = 0 #에이전트 인덱스를 0으로 설정한다. 멀티 에이전트 환경이 아니기 때문이다.

    obs = env.reset() #환경을 리셋하고 초기 관찰을 얻는다.
    rnn_states = torch.zeros(
        [env.num_agents, get_hidden_size(cfg)], 
        dtype=torch.float32, 
        device=device
    ) #각 에이전트의 RNN 상태를 초기화한다.

    while num_frames < max_num_frames and num_episodes < target_num_episodes: #최대 프레임 수와 목표 에피소드 수 전까지 루프를 계속한다.
        with torch.no_grad(): #그래디언트 계산 없이 연산을 수행하도록 설정한다.
            obs_torch = AttrDict(transform_dict_observations(obs)) #관찰값을 변환하고 AttrDict로 감싼다.

            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float() #각 관찰값을 텐서로 변환하고 설정한 장치로 이동한다.

            actions, rnn_states = mt_model.get_outputs(obs_torch, rnn_states)
            actions = actions.cpu().numpy()

            obs, rew, done, infos = env.step(actions) #선택한 행동으로 환경을 진행하고, 새로운 관찰값, 보상, 완료 여부, 추가 정보를 얻는다.

            reward_obs = obs[0].copy() #수정될 수 있는 관찰값을 복사한다.

            #보상 모델을 사용하여 내재적 보상을 계산한다.
            int_reward = reward_model(
                reward_obs, 
                add_dim=True
            ).rewards.cpu().numpy()[0][0]
            msg = bytes(obs[0]['message']) #현재 관찰값에서 메시지를 바이트로 변환
            norm_int_reward = (int_reward - mean) / (var)**(1/2) #내재적 보상을 정규화한다.

            if msg not in episode_msgs: #메시지가 새로운 경우
                episode_msgs[msg] = (1, int_reward, norm_int_reward) #메시지 카운트, 내재적 보상, 정규화된 내재적 보상을 추가한다.
            else: #메시지가 이미 존재하는 경우
                #메시지 카운트를 증가시키고 보상 정보를 유지한다.
                count, int_r, norm_int_r = episode_msgs[msg]
                episode_msgs[msg] = (count+1, int_r, norm_int_r) 

            episode_reward += rew #현재 보상을 에피소드 보상에 추가한다.
            num_frames += 1
            episode_num_frames += 1

            if done[agent_i]: #에피소드가 완료되었는지 확인
                finished_episode[agent_i] = True
                episode_rewards[agent_i].append(episode_reward[agent_i]) #에피소드 보상을 저장한다.

                print("\n" * 29)
                print("===============Top messages=================")
                #현재 에피소드의 상위 100개 메시지를 정렬한다.
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

                #변수 초기화
                episode_msgs = {}
                episode_num_frames = 0
                episode_reward[agent_i] = 0
                num_episodes += 1
                rnn_states[agent_i] = torch.zeros(
                    [get_hidden_size(cfg)], 
                    dtype=torch.float32, 
                    device=device
                ) #RNN 상태를 초기화한다.
                input("Press 'Enter' to continue...")

            if cfg.render: #렌더링 설정이 활성화된 경우
                #현재 타임스텝 통계(프레임 수, 보상, 총 보상, 내재적 보상, 정규화된 내재적 보상)를 출력한다.
                print(f"Timestep: {num_frames} Reward: {rew[0]:.3f} "
                    f"Return: {episode_reward[0]:.3f} "
                    f"Intrinsic reward: {int_reward:.3f} "
                    f"Norm Intrinsic reward: {norm_int_reward:.3f}")

                #환경을 렌더링한다.
                env.render()

                #출력 화면을 위로 27줄 이동시킨다.
                print("\033[%dA" % 28)

                #설정된 시간만큼 렌더링을 일시 중지한다.
                time.sleep(cfg.sleep)

    env.close() #환경을 종료한다.

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards) #함수 실행 결과로 성공 상태와 에피소드 보상의 평균을 반환한다.


def add_extra_params(parser): #parser는 명령행 인자를 파싱하기 위한 객체
    """
    RL 에이전트를 시각화하기 위한 추가 명령행 인자를 지정
    """
    p = parser
    p.add_argument("--sleep", default=0.0, type=float, 
                   help="Controls the speed at which rendering happens.") #--sleep 인자를 추가한다. 렌더링 속도를 제어하는 옵션이다.
    p.add_argument("--render", default=True, type=str2bool, 
                   help="To render or not the game.") #--render 인자를 추가한다. 게임을 렌더링할지 여부를 설정하는 옵션이다.


def parse_all_args(argv=None, evaluation=True): #argv는 명령행 인자, evaluation은 평가 모드 여부
    parser = arg_parser(argv, evaluation=evaluation) #명령행 인자를 파싱하기 위한 parser 객체를 생성한다.
    add_extra_params(parser) #이전에 정의한 add_extra_params 함수를 호출하여 추가 인자를 parser에 추가한다.
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser) #parse_args 함수로 실제 인자를 파싱하고, 결과를 cfg에 저장한다.
    return cfg #파싱된 설정을 반환한다.


def main():
    """평가를 위한 진입점"""
    cfg = parse_all_args() #parse_all_args 함수를 호출하여 설정을 파싱하고, 결과를 cfg 변수에 저장한다.
    _ = enjoy(cfg) #enjoy 함수를 호출하여 파싱된 설정을 사용해 RL 에이전트를 실행한다. 반환값은 사용하지 않기 때문에 _로 저장한다.
    return


if __name__ == '__main__':
    sys.exit(main())
