import torch
import numpy as np
import random
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic

class MultiTaskAgent :
    def __init__(self, cfg, env) :
        self.models = np.empty(5, dtype=object)
        self.env = env
        self.cfg = cfg
        self.get_models()

        self.policy_outputs = np.empty(5, dtype=object)
        self.values =np.zeros(5)
        self.rnn_states = np.empty(5, dtype=object)

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
            self.models[i] = self.load_model(self.cfg, i+1)
            print(i+1, " model load")
    
    def forward_models(self, obs_torch, rnn_states) :
        for i in range(5) :
            self.policy_outputs[i] = self.models[i](obs_torch, rnn_states[i])
        return self.policy_outputs

    #load calibrating mlp

    #calculate calibrated reward

    #task selector
    def select_task(self, policy_outputs) :
        for i in range(5) :
            self.values[i] = policy_outputs[i].values.item()
            self.rnn_states[i] = policy_outputs[i].rnn_states

        actions = policy_outputs[np.argmax(self.values)].actions
        return actions, self.rnn_states

    def get_outputs(self, obs_torch, rnn_states) :
        return self.select_task(self.forward_models(obs_torch, rnn_states))
