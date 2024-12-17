import torch
import numpy as np
from calibrator import create_calibrator
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic

class MultiTaskAgent :
    def __init__(self, cfg, env) :
        self.models = np.empty(5, dtype=object)
        self.env = env
        self.cfg = cfg
        self.get_models()

        self.policy_outputs = np.empty(5, dtype=object)
        self.rnn_states = np.empty(5, dtype=object)

        self.new_distributions = np.empty(5)
        self.corrected_rate = np.empty(5)
        self.calibrators = np.empty(5, dtype=object)

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
        for i in range(5) :
            self.models[i] = self.load_model(self.cfg, i)
            print(i, " model load")
    
    def forward_models(self, obs_torch, rnn_states) :
        for i in range(5) :
            self.policy_outputs[i] = self.models[i](obs_torch, rnn_states[i])
        return self.policy_outputs

    #load calibrating mlp
    def load_calibrator(self, task_number):
        device = torch.device('cpu' if self.cfg.device == 'cpu' else 'cuda')
        for i in range(task_number):
            calibrator = create_calibrator(self.cfg, self.env.observation_space)
            calibrator.load_state_dict(torch.load(f'train_dir/calibrator_dir/c{i}/checkpoint_p{0}.pth', map_location=device))
            calibrator.to(device)
            self.calibrators[i] = calibrator
            print(f"Calibrator {i} loaded")

    def calculate_corrected_rates(self, obs_torch, task_number):
        self.corrected_rate = np.empty(task_number, dtype=object)
        for i in range(task_number):
            with torch.no_grad():
                hypothesis = self.calibrators[i](obs_torch)
                corrected_distribution = torch.softmax(hypothesis, dim=-1)
                self.corrected_rate[i] = corrected_distribution

    #task selector - modified
    def select_task(self, policy_outputs, obs_torch) :
        max_idx = -1
        max_value = float('-inf')
        task_number = 2 #task 갯수

        #mlp불러오는 코드 추가, 보정률 담기
        self.load_calibrator()
        self.calculate_corrected_rates(obs_torch, task_number)

        #분포에 보정률 곱하기
        for i in range(task_number) :
            original_probs = policy_outputs[i].action_distribution.probs
            self.new_distributions[i] = torch.mul(original_probs, self.corrected_rate[i])
            self.rnn_states[i] = policy_outputs[i].rnn_states

        #보정률을 곱한 분포에서 각 최댓값의 index 찾기
        for i, tensor in enumerate(self.new_distributions) :
            curr_value = torch.max(tensor).item()
            curr_idx = torch.argmax(tensor).item()

            if curr_value > max_value:
                max_value = curr_value
                max_idx = int(curr_idx)
        
        #tensor로 변환하여 행동에 넣기
        actions = torch.tensor([max_idx], device='cuda:0')
        return actions, self.rnn_states

    def get_outputs(self, obs_torch, rnn_states) :
        return self.select_task(self.forward_models(obs_torch, rnn_states))
