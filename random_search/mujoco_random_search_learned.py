from random_search.online_learner import OnlineLearner
from random_search.sampler import Sampler 
from random_search.ars_interface import RayWorkerForMujoco

import os
import time
import datetime
import ray
import torch
import numpy as np
from random_search.policies import *
from tensorboardX import SummaryWriter

class MujocoRandomSearchLearned(object):
    def __init__(self, params):
        valid_keys =  ["sampler","env_name", "seed", "dir_std", "step_size", "num_rollouts", "top_k"]
        exp_identifier = '___'.join(['{}_{}'.format(key,val) for (key,val) in params.items() if key in valid_keys])
        self.writer = SummaryWriter(log_dir='runs/{}_{}'.format(exp_identifier, datetime.datetime.now().strftime("%I%M%p_%B%d")))
 
        if params["sampler"] in ['sgd']:
            print("SGD is Enabled")
            self.sgd = True
            params["sampler"] = "unit_normal"
        else:
            self.sgd = False 

        if params["sampler"] in ['fixed_subspace']:
            print('Fixed Subspace is Enabled')
            self.fixed_subspace = True 
            params["sampler"] = "jacobian_normalize"
        else:
            self.fixed_subspace = False

        self.dir_std = params["dir_std"]
        self.num_directions = params["num_rollouts"]
        self.dimension = params['dimension']

        self.policy = LinearPolicy(params["policy_params"])
        self.current_solution = self.policy.get_weights()

        self.step_size = params["step_size"]
        self.train_stats = []
        self.eval_stats = []
 
        ray.init(num_cpus=params["num_workers"], include_webui=False, ignore_reinit_error=True)
        self.num_workers = params["num_workers"]
        self.workers = [RayWorkerForMujoco.remote(params["seed"] + 42 * i, params = params) for i in range(params["num_workers"])]

        self.sampler_type =  params["sampler"]
        self.effective_dimension = self.dimension
        if params["filter_corrected"]:
            self.effective_dimension += self.policy.ac_dim
            self.filter_corrected = True
        else:
            self.filter_corrected = False

        self.old_gradients = []

        if not "unit_normal" in self.sampler_type or self.sgd:
            print('Sampler is {}; so a learner is created'.format(self.sampler_type))
            self.learner = OnlineLearner(params['optimizer'], params['learning_rate'], self.filter_corrected,
                                         self.dimension, self.effective_dimension, params['num_hidden_dim'], 
                                         params['num_learning_iterations'], make_independent=params['gram_schmidt'], cuda=False)
            if self.filter_corrected:
                self.learner.filters = self.policy.get_weights_plus_stats()
        else:
            self.learner = None
        self.sampler = Sampler(num_directions=params["num_rollouts"], dimension=self.dimension, K=50, dir_std=params["dir_std"], sampler_type=self.sampler_type, learner=self.learner, GPU=False)  

        # Variance reduction is not open sourced, contuct us if you are interested
        self.variance_reduced = False
        self.ts = 0
        if self.learner:
            self.writer.add_scalar('hyper_params_lr', params['learning_rate'], 1)
            self.writer.add_scalar('hyper_params_num_hidden_dim', params['num_hidden_dim'], 1)
            self.writer.add_scalar('hyper_params_num_learning_iterations', params['num_learning_iterations'], 1)
        torch.manual_seed(params['seed'])
        np.random.seed(params['seed'])
        self.model_folder = 'saved_models/{}_{}/'.format(exp_identifier, datetime.datetime.now().strftime("%I%M%p_%B%d"))
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        
        if 'top_k' in params:
            self.top_k = params['top_k']
        else:
            self.top_k = params['num_rollouts']
        print('TOP-K is {}'.format(self.top_k))

 
    def update_models(self, current_sol, directions, y):
        sol_torch = torch.from_numpy(current_sol).view(1,-1).float()

        if self.fixed_subspace: 
            self.sampler.current_solution = sol_torch
            self.learner.filters = self.policy.get_weights_plus_stats()
            return 

        if self.filter_corrected:
            self.learner.update_filter_corrected(sol_torch, directions, self.policy.get_weights_plus_stats(), y)
        else:
            self.learner.update(sol_torch, directions, y)
        self.sampler.current_solution = sol_torch
        self.learner.filters = self.policy.get_weights_plus_stats()

    def rollouts_direct(self, directions):
        directions_n = directions.numpy()
        rollout_per_worker = int(directions_n.shape[0]/self.num_workers)
        # Current implementation is incomplete and only support this
        assert(rollout_per_worker*self.num_workers == directions_n.shape[0])

        # Sync all workers first
        current_policy = ray.put(self.current_solution)

        # Do rollouts
        rollouts = []
        for rollout in range(rollout_per_worker):
            for worker in range(self.num_workers):
                perturbation_for_worker = ray.put(directions_n[worker+rollout*self.num_workers])
                rollouts+= [self.workers[worker].do_rollouts_c_policy.remote(current_policy, perturbation_for_worker, evaluate=False)]

        results = ray.get(rollouts)

        pos_rollouts = []
        time = 0
        for result in results:
            time += result['steps']
            pos_rollouts += [result['rollout_rewards'][0]['+']]


        return torch.from_numpy(np.array(pos_rollouts)).view(directions_n.shape[0], 1).float(), time

    def rollouts(self, directions):
        directions_n = directions.numpy()
        rollout_per_worker = int(directions_n.shape[0]/self.num_workers)
        # Current implementation is incomplete and only support this
        assert(rollout_per_worker*self.num_workers == directions_n.shape[0])

        # Sync all workers first
        current_policy = ray.put(self.current_solution)

        # Do rollouts
        rollouts = []
        for rollout in range(rollout_per_worker):
            for worker in range(self.num_workers):
                perturbation_for_worker = ray.put(directions_n[worker+rollout*self.num_workers])
                rollouts+= [self.workers[worker].do_rollouts_same_policy.remote(current_policy, perturbation_for_worker, evaluate=False)]

        results = ray.get(rollouts)

        pos_rollouts_un = []
        neg_rollouts_un = []
 
        pos_rollouts = []
        neg_rollouts = []
        time = 0
        for result in results:
            time += result['steps']
            pos_rollouts += [result['rollout_rewards'][0]['+']]
            neg_rollouts += [result['rollout_rewards'][0]['-']]

            pos_rollouts_un += [result['rollout_rewards'][0]['un_+']]
            neg_rollouts_un += [result['rollout_rewards'][0]['un_-']]


        return {'+': torch.from_numpy(np.array(pos_rollouts)).view(directions_n.shape[0], 1).float(), 
                '-': torch.from_numpy(np.array(neg_rollouts)).view(directions_n.shape[0], 1).float(),
                'un_+': torch.from_numpy(np.array(pos_rollouts_un)).view(directions_n.shape[0], 1).float(), 
                'un_-': torch.from_numpy(np.array(neg_rollouts_un)).view(directions_n.shape[0], 1).float()
                }, time

    def update_sgd(self, rewards, directions):
        # This is  rather different and kind of hard-core version
        mx_rewards = torch.max(rewards['+'], rewards['-'])
        ss, ind = torch.sort(mx_rewards, dim=0, descending=True)
        chosen_indices = ind[0:self.top_k,0]

        rewards['+'] = rewards['+'][chosen_indices,:]
        rewards['-'] = rewards['-'][chosen_indices,:]
        stddev = torch.std(torch.cat((rewards['+'], rewards['-']),0), unbiased=False)

        gd = self.learner.get_derivative_at_x(self.sampler.current_solution)
        gd = gd / stddev
        update = self.step_size * gd
        
        self.current_solution += update.numpy().reshape(self.current_solution.shape)
        self.update_models(self.current_solution, directions, rewards)


    def update(self, rewards, directions):
        # This is  rather different and kind of hard-core version
        mx_rewards = torch.max(rewards['+'], rewards['-'])
        ss, ind = torch.sort(mx_rewards, dim=0, descending=True)
        chosen_indices = ind[0:self.top_k,0]

        directions = directions[chosen_indices,:]
        rewards['+'] = rewards['+'][chosen_indices,:]
        rewards['-'] = rewards['-'][chosen_indices,:]
        rewards['un_+'] = rewards['un_+'][chosen_indices,:]
        rewards['un_-'] = rewards['un_-'][chosen_indices,:]

        stddev = torch.std(torch.cat((rewards['+'], rewards['-']),0), unbiased=False)
        print(stddev)
        if stddev < 1:
            stddev = 1
        rewards['+'] /= stddev
        rewards['-'] /= stddev
        directional_grads = rewards['+'] - rewards['-']
 
        final_direction = torch.matmul(directions.transpose(0,1), directional_grads)
        final_direction = final_direction / directions.shape[0]
        final_direction = final_direction / self.dir_std 

        if len(self.old_gradients) > 5:
            self.old_gradients.pop(0)
        self.old_gradients.append(final_direction)

        update = self.step_size * final_direction
        self.current_solution += update.numpy().reshape(self.current_solution.shape)
        
        if not "unit_normal" in self.sampler_type:
            self.update_models(self.current_solution, directions, rewards)

        stddev2 = torch.std(torch.cat((rewards['un_+'], rewards['un_-']),0), unbiased=False)
        print(stddev2)
        if stddev2 < 1:
            stddev2 = 1
        rewards['un_+'] /= stddev2
        rewards['un_-'] /= stddev2
 
    def stats_train_direct(self, episodes, rewards):
        top = torch.max(rewards).item()
        bottom = torch.min(rewards).item()
        average = torch.mean(rewards).item() 
        self.train_stats.append({'Iter':episodes, 'Max':top, 'Min':bottom, 'Mean': average})
        self.writer.add_scalar('train_average', average, self.ts)
        self.writer.add_scalar('train_max', top, self.ts)
        self.writer.add_scalar('train_min', bottom, self.ts)
        self.writer.add_scalar('train_episodes', episodes, self.ts)

    def stats_train(self, episodes, rewards):
        top = max(torch.max(rewards['+']).item(), torch.max(rewards['-']).item())
        bottom = min(torch.min(rewards['+']).item(), torch.min(rewards['-']).item())
        average = (torch.mean(rewards['+']).item() + torch.mean(rewards['-']).item())/2
        self.train_stats.append({'Iter':episodes, 'Max':top, 'Min':bottom, 'Mean': average})
        self.writer.add_scalar('train_average', average, self.ts)
        self.writer.add_scalar('train_max', top, self.ts)
        self.writer.add_scalar('train_min', bottom, self.ts)
        self.writer.add_scalar('train_episodes', episodes, self.ts)

    def post_iteration_cleanup(self):
        # Collect all stats 
        for i in range(self.num_workers):
            self.policy.observation_filter.update(ray.get(self.workers[i].get_filter.remote()))
        self.policy.observation_filter.stats_increment()
        self.policy.observation_filter.clear_buffer()

        filter = ray.put(self.policy.observation_filter)
        sync_filters = [worker.sync_filter.remote(filter) for worker in self.workers]
        # Wait for the sync
        ray.get(sync_filters)

        increment_filters = [worker.stats_increment.remote() for worker in self.workers]
        # waiting for increment of all workers
        ray.get(increment_filters)         

    def evaluate(self, num_episodes):
        policy_id = ray.put(self.current_solution)
        rollout_per_worker = int(50/self.num_workers) + 1

        rollouts = [ worker.do_rollouts_same_policy.remote(policy_id, None,
                                               num_rollouts=rollout_per_worker, 
                                               evaluate=True) for worker in self.workers]

        results = ray.get(rollouts)

        rewards = []
        for result in results:
            rewards += result["rollout_rewards"]

        rewards = np.array(rewards, dtype=np.float64)
        self.eval_stats.append({'AverageRewards': np.mean(rewards), 'StdRewards': np.std(rewards), 
                'MaxRewards': np.max(rewards), 'MinRewards': np.min(rewards)})
        print(self.eval_stats[-1])
        self.writer.add_scalar('evaluation_average_reward', self.eval_stats[-1]['AverageRewards'], self.ts)
        self.writer.add_scalar('evaluation_std_reward', self.eval_stats[-1]['StdRewards'], self.ts)
        self.writer.add_scalar('evaluation_min_reward', self.eval_stats[-1]['MaxRewards'], self.ts)
        self.writer.add_scalar('evaluation_max_reward', self.eval_stats[-1]['MinRewards'], self.ts)

        PATH_T = self.model_folder + 'learner_{}'.format(self.ts)
        PATH_M = self.model_folder + 'policy_{}'.format(self.ts)
        if self.learner:
            torch.save(self.learner.model.state_dict(), PATH_T)
        current_policy_to_save = self.policy.get_weights_plus_stats()
        np.savez(PATH_M, current_policy_to_save)


    def search(self, max_iteration, validation_epoch):
        num_episodes = 0
        self.sampler.current_solution = torch.from_numpy(self.current_solution).view(1,-1).float()
        
        if self.fixed_subspace:
            print('loading')
            self.learner.model.load_state_dict(torch.load('learner_ant'))


        if self.sampler_type in ['cma_es']:
            import cma
            print(self.current_solution.reshape(self.dimension))
            es = cma.CMAEvolutionStrategy( (self.current_solution.reshape(self.dimension)).tolist(), self.dir_std, {'popsize': self.num_directions})
        elif self.sampler_type in ['rembo']:
            from bayesian_optimization_interface import BOInterFace
            boi = BOInterFace(self.num_directions, 25, 100)

        for iteration in range(max_iteration):
            self.ts = iteration + 1
            if iteration % validation_epoch == 0:
                # Evaluate at every validation_epoch
                #print('Evaluation at {}'.format(num_episodes))
                self.evaluate(num_episodes)
                #print('Results: {}'.format(self.eval_stats[-1]))

            #print('Sampling Directions')
            if self.sampler_type in ['guided_es']:
                directions = self.sampler.sample_guided_es(self.old_gradients)
            elif self.sampler_type in ['cma_es']:
                directions_c = es.ask()
                directions = torch.from_numpy(np.array(directions_c))
            elif self.sampler_type in ['rembo']:
                directions_c = boi.acquire()
                directions = torch.from_numpy(np.array(directions_c))
            else:
                directions = self.sampler.sample()
            #print(directions)

            #print('Rollouts')
            if self.sampler_type in ['cma_es']:
                rewards, num_eval = self.rollouts_direct(directions)
            elif self.sampler_type in ['rembo']:
                rewards, num_eval = self.rollouts_direct(directions)                
            else:
                rewards, num_eval = self.rollouts(directions)
            #print(rewards)
            #print(num_eval)
            num_episodes+=num_eval

            if self.sampler_type in ['cma_es']:
                self.stats_train_direct(num_episodes, rewards)
            else:
                self.stats_train(num_episodes, rewards)
 
            #print('Training Results: {}'.format(self.train_stats[-1]))

            if self.sgd:
                self.update_sgd(rewards, directions)
            elif self.variance_reduced:
                raise ValueError("Variance reduced version i not shared yet!")
            elif self.sampler_type in ['cma_es']:
                flat_rewards = [(-1.0)*v[0] for v in (rewards).tolist()]
                es.tell(directions_c, flat_rewards)
                self.current_solution = np.array(es.result.xbest).reshape(self.current_solution.shape)
            else:
                self.update(rewards, directions)

            self.post_iteration_cleanup()
            #print('Current Solution is: {}'.format(self.current_solution))
            #print(self.sampler.current_solution)


        return self.train_stats, self.eval_stats
