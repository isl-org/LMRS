"""
Modified from https://github.com/modestyachts/ARS/blob/master/code/ars.py
"""
import ray
import time
import datetime
import gym
import numpy as np
from random_search.policies import *

@ray.remote
class RayWorkerForMujoco(object):
    def __init__(self, env_seed, params, deltas=None):
        # Make the Gym environment in each worker for parallel eval
        self.env = gym.make(params['env_name'])
        self.env.seed(env_seed)

        self.policy = LinearPolicy(params["policy_params"])
        self.shift = params["shift"]
        self.rollout_length = params["rollout_length"]
        
    def rollout(self, shift = 0.0, rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.0
        total_unshifted_reward = 0.0
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            total_unshifted_reward += reward
            if done:
                break
            
        return total_reward, steps, total_unshifted_reward  

    def do_rollouts_c_policy(self, w_policy, perturbation, num_rollouts = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        rollout_rewards, deltas_idx, direction_weights = [], [], []
        steps = 0

        self.policy.update_weights(w_policy)
        for i in range(num_rollouts):
            if evaluate:                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps, _ = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                rollout_rewards.append(reward)
            else:
                delta = perturbation.reshape(w_policy.shape)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(delta)
                pos_reward, pos_steps, pos_reward_un  = self.rollout(shift = self.shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                steps += pos_steps

                rollout_rewards.append({'+': pos_reward_un})
                            

        return {'rollout_rewards': rollout_rewards, "steps" : steps}


    def do_rollouts_same_policy(self, w_policy, perturbation, num_rollouts = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """
        rollout_rewards, deltas_idx, direction_weights = [], [], []
        steps = 0

        self.policy.update_weights(w_policy)
        for i in range(num_rollouts):
            if evaluate:                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps, _ = self.rollout(shift = 0., rollout_length = self.env.spec.timestep_limit)
                rollout_rewards.append(reward)
            else:
                delta = perturbation.reshape(w_policy.shape)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps, pos_reward_un  = self.rollout(shift = self.shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps, neg_reward_un = self.rollout(shift = self.shift)
                steps += pos_steps + neg_steps

                rollout_rewards.append({'+': pos_reward, '-': neg_reward, 
                                        'un_+': pos_reward_un, 'un_-': neg_reward_un })
                            

        return {'rollout_rewards': rollout_rewards, "steps" : steps}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    def get_weights_plus_stats(self):
        return self.policy.get_weights_plus_stats()
