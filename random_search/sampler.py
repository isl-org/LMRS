import torch
import numpy as np


class Sampler(object):
    def __init__(self, num_directions, dimension, K, dir_std, sampler_type, learner, GPU):
        self.num_directions = num_directions
        self.dimension = dimension
        self.K = K
        self.learner = learner
        self.gpu = GPU
        self.dir_std = dir_std

        self.sampler_type = sampler_type
        
        valid_samplers = ['cma_es','guided_es', 'unit_normal', 'value', 'angle', 'jacobian', 'jacobian_normalize']
        if self.sampler_type not in valid_samplers:
            raise ValueError('{} is not a valid sampler, valid ones are:{}'.format(self.sampler_type, valid_samplers))


    def sample_guided_es(self, old_grads):
        if len(old_grads) < 1:
            print('first')
            return self.sample_normal_from_torch(self.num_directions, self.gpu)
        grads_after_gs = []
        grads_after_gs.append(old_grads[0])
        for i in range(1, len(old_grads)):
            olo = old_grads[i]
            init_mag = torch.dot(olo.view(-1), olo.view(-1))
            for j in range(len(grads_after_gs)):
                dependent_part = torch.dot(olo.view(-1), grads_after_gs[j].view(-1)) / torch.dot(grads_after_gs[j].view(-1), grads_after_gs[j].view(-1))
                olo = olo - dependent_part * grads_after_gs[j]
            final_mag = torch.dot(olo.view(-1), olo.view(-1)) / init_mag

            if final_mag > 1e-4:
                nrm = np.sqrt(final_mag * init_mag)
                grads_after_gs.append((olo/nrm)*np.sqrt(self.dimension))
        
        if len(grads_after_gs) > 0:
            U = torch.rand(self.num_directions, len(grads_after_gs))
            Up = torch.sum(U, dim=1).view(self.num_directions, 1)
            U = U / Up
            V = torch.zeros(self.num_directions, self.dimension)
            for dire in range(self.num_directions):
                for k in range(len(grads_after_gs)):
                    V[dire, :] += U[dire, k] * grads_after_gs[k].view(V[dire,:].shape)

            alpha = 0.5
            B = 0.5*self.dir_std*torch.rand(self.num_directions, self.dimension)
            B += 0.5*self.dir_std*V
            return B
        else:
            return self.dir_std*torch.rand(num_directions, self.dimension).cuda()



    def sample(self, no=False):
        if no:
            return self.sample_normal_from_torch(self.num_directions, self.gpu)
 
        if self.sampler_type == 'unit_normal':
            return self.sample_normal_from_torch(self.num_directions, self.gpu)
        if self.sampler_type == 'value':
            return self.sample_with_learner_using_value(self.num_directions, self.gpu)
        elif self.sampler_type == 'angle':
            return self.sample_with_learner_using_angle(self.num_directions, self.gpu)
        elif self.sampler_type == 'jacobian':
            return self.sample_with_learner_using_jacobian(False, self.gpu)
        elif self.sampler_type == 'jacobian_normalize':
            return self.sample_with_learner_using_jacobian(True, self.gpu)

    def sample_normal_from_torch(self, num_directions, gpu=True):
        if gpu:
            return self.dir_std*torch.rand(num_directions, self.dimension).cuda()
        else:
            return self.dir_std*torch.rand(num_directions, self.dimension)

    def sample_with_learner_using_value(self, num_directions, gpu=True):
        directions_a = self.sample_normal_from_torch(num_directions*self.K, gpu)
        
        if self.learner.filter_corrected:
            positive_values = self.learner.evaluate_filter_corrected(self.current_solution + directions_a)
            negative_values = self.learner.evaluate_filter_corrected(self.current_solution - directions_a)
        else:
            positive_values = self.learner.evaluate(self.current_solution + directions_a)
            negative_values = self.learner.evaluate(self.current_solution - directions_a)

        if self.gpu:
            values = torch.max(positive_values, negative_values).view(-1).cpu().detach().numpy()
        else:
            values = torch.max(positive_values, negative_values).view(-1).detach().numpy()
        # Convert values to a distribution
        valm = values - np.min(values)
        if np.sum(valm) < 1e-7:
            valm = np.ones(valm.shape)
        valm = valm / np.sum(valm)
        sel = np.random.choice(self.num_directions*self.K, self.num_directions, p=valm)
        directions = directions_a[sel]
        return directions
    
    def sample_with_learner_using_angle(self, num_directions, gpu=True):
        if self.learner.filter_corrected:
            angle = self.learner.get_angle_filter_corrected(self.current_solution)
        else:
            angle = self.learner.get_angle(self.current_solution)

        directions_a = self.sample_normal_from_torch(num_directions*self.K, gpu)
        if self.learner.filter_corrected:
            directions_a_c = self.learner.get_corrected_directions(directions_a)
        else:
            directions_a_c = directions_a

        dps = torch.abs(torch.mm(directions_a_c, angle.view(-1,1)))
        mag = torch.sqrt(torch.sum(directions_a_c*directions_a_c, dim=1)) * torch.sqrt(torch.sum(angle*angle))
        dps = dps / mag.view(-1, 1)

        if gpu:
            values = dps.view(-1).cpu().numpy()
        else:
            values = dps.view(-1).numpy()

        if np.sum(values) < 1e-7:
            values = np.ones(values.shape)
        values = values / np.sum(values)
        sel = np.random.choice(self.num_directions*self.K, self.num_directions, p=values)
        directions = directions_a[sel]
        return directions
    
    def sample_with_learner_using_jacobian(self, normalize, gpu=True):
        if self.learner.filter_corrected:
            if normalize:
                ret, J = self.learner.get_jacobian_corrected(self.current_solution, normalize=True, gpu=gpu)
                if ret:
                    jac = self.dir_std*J
                else:
                    print('really failed fall back to unit normal')
                    return self.sample_normal_from_torch(self.num_directions, gpu)

                self.learner.get_jacobian_corrected(self.current_solution, normalize=True, gpu=gpu)
            else:
                jac = self.learner.get_jacobian_corrected(self.current_solution, normalize=False, gpu=gpu)
        else:
            if normalize:
                jac = self.dir_std*self.learner.get_jacobian(self.current_solution, normalize=True, gpu=gpu)
            else:
                jac = self.learner.get_jacobian(self.current_solution, normalize=False, gpu=gpu)

        num_dir = jac.shape[0]
        directions = jac
        if num_dir > self.num_directions:
            selected_dir = np.random.choice(num_dir, self.num_directions)
            directions = jac[selected_dir]
        if num_dir < self.num_directions:
            if gpu:
                extra_dir = self.dir_std*torch.randn(self.num_directions - num_dir, self.dimension).cuda()
            else:
                extra_dir = self.dir_std*torch.randn(self.num_directions - num_dir, self.dimension)

            directions = torch.cat((extra_dir, jac), dim=0)
        return directions
