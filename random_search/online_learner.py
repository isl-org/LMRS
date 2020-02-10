import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
from random_search.synthetic_environment import SyntheticFunctionLearner

class OnlineLearner(object):
    def __init__(self, opt, filter_corrected, lrate, dim, effective_dim, hidden, num_iter, make_independent, cuda=True):
        self.hidden = hidden
        self.dimension = dim
        self.effective_dim = effective_dim
        self.filter_corrected = filter_corrected
        if self.filter_corrected:
            self.filters = None

        self.model = SyntheticFunctionLearner(effective_dim, hidden, 1)
        if cuda:
            self.model.cuda()
        if opt=='sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lrate, momentum=0.9)
        elif opt=='adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lrate)
        else:
            raise ValueError('Optimizer is not recognized')

        self.num_iteration = num_iter

        self.make_independent = make_independent

        self.oracle = False
        self.losses = []
        
        print('Learner with {}->{}->1 using {} with gs:{}'.format(dim, hidden,self.optimizer,self.make_independent))

    def set_model(self, new_model):
        print('Using pre-trained model, so no Online Learning will happen')
        self.model.load_state_dict(new_model.state_dict())
        self.oracle = True

    def update_filter_corrected(self, cur_sol, directions, filters, y):
        pw = filters[0]
        mu = filters[1]
        std = filters[2]
        std[std<1e-5] = 1
        
        pos_dir = cur_sol + directions
        neg_dir = cur_sol - directions

        pos_batch = pos_dir.view(directions.shape[0], pw.shape[0], pw.shape[1])
        pos_weights = torch.matmul(pos_batch, torch.from_numpy(np.diag(1/std)).float())
        pos_biases = -1.0 * torch.matmul(pos_weights, torch.from_numpy(mu).float())
        final_pos = torch.cat((pos_weights.view(directions.shape[0], pw.shape[0]*pw.shape[1]), pos_biases), dim = 1)

        neg_batch = neg_dir.view(directions.shape[0], pw.shape[0], pw.shape[1])
        neg_weights = torch.matmul(neg_batch, torch.from_numpy(np.diag(1/std)).float())
        neg_biases = -1.0 * torch.matmul(neg_weights, torch.from_numpy(mu).float())
        final_neg = torch.cat((neg_weights.view(directions.shape[0], pw.shape[0]*pw.shape[1]), neg_biases), dim = 1)

        all_dir = torch.cat((final_pos, final_neg), dim=0)
        all_y = torch.cat((y['+'], y['-']), dim=0)

        all_dir_v = all_dir.detach()
        all_y_v = all_y.detach()

        for i in range(self.num_iteration):
            self.optimizer.zero_grad()

            outputs, _ = self.model(all_dir_v)
            lss = torch.mean(torch.abs(outputs - all_y_v))
            if i < 1:
                self.losses.append(lss.item())
            lss.backward()
            self.optimizer.step()

    def update(self, cur_sol, directions, y):
        if self.oracle:
            return
        pos_dir = cur_sol + directions
        neg_dir = cur_sol - directions
        all_dir = torch.cat((pos_dir, neg_dir), dim=0)
        all_y = torch.cat((y['un_+'], y['un_-']), dim=0)

        all_dir_v = all_dir.detach()
        all_y_v = all_y.detach()

        for i in range(self.num_iteration):
            self.optimizer.zero_grad()

            outputs, _ = self.model(all_dir_v)
            lss = torch.mean(torch.abs(outputs - all_y_v))
            if i < 1:
                self.losses.append(lss.item())
            lss.backward()
            self.optimizer.step()

    def get_jacobian(self, X, normalize=True, gpu=True):
        grads = []
        X = X.view(1, -1).detach()
        XV = Variable(X, requires_grad=True)

        if gpu:
            ohot = torch.eye(self.hidden).cuda()
        else:
            ohot = torch.eye(self.hidden)

        for i in range(self.hidden):
            self.optimizer.zero_grad()
            out, hidden = self.model(XV)
            s = torch.sum(hidden*ohot[i,:])
            s.backward()
            grads.append(XV.grad.detach().clone())

        grads_after_gs = []
        
        if normalize:
            gs_tresh = 1e-5
        else:
            gs_tresh = 1e-3
       
        grads_after_gs.append(grads[0])
        if self.make_independent:
            for i in range(1,self.hidden):
                init_mag = torch.dot(grads[i].view(-1), grads[i].view(-1))
                for j in range(len(grads_after_gs)):
                    dependent_part = torch.dot(grads[i].view(-1), 
                                               grads_after_gs[j].view(-1)) / torch.dot(grads_after_gs[j].view(-1), 
                                                                                       grads_after_gs[j].view(-1))
                    grads[i] = grads[i] - dependent_part * grads_after_gs[j]
                final_mag =  torch.dot(grads[i].view(-1), grads[i].view(-1)) / init_mag
                if final_mag > gs_tresh:
                    grads_after_gs.append(grads[i])
        else:
            grads_after_gs = grads

        J = torch.cat(grads_after_gs, dim=0)
        
        if normalize:
            Jsum = torch.sum(J*J, dim=1, keepdim=True)
            J = J / torch.sqrt(Jsum)
            J = J * (np.sqrt(self.dimension))
        return J

    def get_jacobian_corrected(self, X, normalize=True, gpu=True):
        pw = self.filters[0]
        mu = self.filters[1]
        std = self.filters[2]
        std[std<1e-5] = 1

        X_mat = X.view(1, pw.shape[0], pw.shape[1])
        X_w = torch.matmul(X_mat, torch.from_numpy(np.diag(1/std)).float())
        X_biases = -1.0 * torch.matmul(X_w, torch.from_numpy(mu).float())
        final_X = torch.cat((X_w.view(1, pw.shape[0]*pw.shape[1]), X_biases), dim = 1)
        final_X = final_X.detach()


        grads = []
        XV = Variable(final_X, requires_grad=True)

        if gpu:
            ohot = torch.eye(self.hidden).cuda()
        else:
            ohot = torch.eye(self.hidden)

        for i in range(self.hidden):
            self.optimizer.zero_grad()
            out, hidden = self.model(XV)
            #print(hidden)
            #print(out)
            s = torch.sum(hidden*ohot[i,:])
            s.backward()
            grads.append(XV.grad.detach().clone())

        grads_after_gs = []
        
        if normalize:
            gs_tresh = 1e-5
        else:
            gs_tresh = 1e-3
       
        init_vec = -1
        for i in range(self.hidden):
            if torch.dot(grads[i].view(-1), grads[i].view(-1)) > 1e-5:
                init_vec = i
                break
        if init_vec < 0:
            print('fail')
            self.model.fc1.reset_parameters()
            self.model.fc2.reset_parameters()
            self.model.fc3.reset_parameters()
            print('reset everything')
            return False, None

        grads_after_gs.append(grads[init_vec])
        if self.make_independent:
            for i in range(init_vec+1,self.hidden):
                init_mag = torch.dot(grads[i].view(-1), grads[i].view(-1))
                for j in range(len(grads_after_gs)):
                    dependent_part = torch.dot(grads[i].view(-1), 
                                               grads_after_gs[j].view(-1)) / torch.dot(grads_after_gs[j].view(-1), 
                                                                                       grads_after_gs[j].view(-1))
                    grads[i] = grads[i] - dependent_part * grads_after_gs[j]
                final_mag =  torch.dot(grads[i].view(-1), grads[i].view(-1)) / init_mag
                if final_mag > gs_tresh:
                    grads_after_gs.append(grads[i])
        else:
            grads_after_gs = grads

        J = torch.cat(grads_after_gs, dim=0)
        
        J = J[:, 0:pw.shape[0]*pw.shape[1]]
        # Correct it back
        J = torch.matmul(J.view(-1, pw.shape[0], pw.shape[1]), torch.from_numpy(np.diag(std)).float())
        J = J.view(-1, pw.shape[0]*pw.shape[1])
        if normalize:
            Jsum = torch.sum(J*J, dim=1, keepdim=True)
            #print('Jsum', Jsum)
            #print(J.shape)
            J = J / torch.sqrt(Jsum)
            J = J * (np.sqrt(self.dimension))
        #print(J)
        return True, J

    def get_angle(self, X):
        X = X.view(1, -1).detach()
        XV = Variable(X, requires_grad=True)

        self.optimizer.zero_grad()
        outputs, hidden = self.model(XV)
        outputs.backward()
        g = XV.grad.detach()
        return g
    
    def get_corrected_directions(self, dirs):
        pw = self.filters[0]
        mu = self.filters[1]
        std = self.filters[2]
        std[std<1e-5] = 1

        dirs_ = dirs.view(dirs.shape[0], pw.shape[0], pw.shape[1])
        dirs_w = torch.matmul(dirs_, torch.from_numpy(np.diag(1/std)).float())
        dirs_b = -1.0 * torch.matmul(dirs_w, torch.from_numpy(mu).float())
        dirs_f = torch.cat((dirs_w.view(dirs.shape[0], pw.shape[0]*pw.shape[1]), dirs_b), dim = 1)
        return dirs_f

    def get_derivative_at_x(self, X):
        X = X.view(1, -1).detach()
        pw = self.filters[0]
        mu = self.filters[1]
        std = self.filters[2]
        std[std<1e-5] = 1

        X_mat = X.view(1, pw.shape[0], pw.shape[1])
        X_w = torch.matmul(X_mat, torch.from_numpy(np.diag(1/std)).float())
        X_biases = -1.0 * torch.matmul(X_w, torch.from_numpy(mu).float())
        final_X = torch.cat((X_w.view(1, pw.shape[0]*pw.shape[1]), X_biases), dim = 1)
        final_X = final_X.detach()

        XV = Variable(final_X, requires_grad=True)
        self.optimizer.zero_grad()
        outputs, hidden = self.model(XV)
        outputs.backward()
        g = XV.grad.detach()

        relevant_g = g[0,0:pw.shape[0]*pw.shape[1]].view(1, pw.shape[0], pw.shape[1])
        relevant_g = torch.matmul(relevant_g, torch.from_numpy(np.diag(1/std)).float())
        relevant_g = relevant_g.view(1, pw.shape[0]*pw.shape[1])
 
        return relevant_g


    def get_angle_filter_corrected(self, X):
        X = X.view(1, -1).detach()
        pw = self.filters[0]
        mu = self.filters[1]
        std = self.filters[2]
        std[std<1e-5] = 1

        X_mat = X.view(1, pw.shape[0], pw.shape[1])
        X_w = torch.matmul(X_mat, torch.from_numpy(np.diag(1/std)).float())
        X_biases = -1.0 * torch.matmul(X_w, torch.from_numpy(mu).float())
        final_X = torch.cat((X_w.view(1, pw.shape[0]*pw.shape[1]), X_biases), dim = 1)
        final_X = final_X.detach()

        XV = Variable(final_X, requires_grad=True)
        self.optimizer.zero_grad()
        outputs, hidden = self.model(XV)
        outputs.backward()
        g = XV.grad.detach()
        return g

    def evaluate_filter_corrected(self, X):
        pw = self.filters[0]
        mu = self.filters[1]
        std = self.filters[2]
        std[std<1e-5] = 1
         
        X_mat = X.view(-1, pw.shape[0], pw.shape[1])
        X_w = torch.matmul(X_mat, torch.from_numpy(np.diag(1/std)).float())
        X_biases = -1.0 * torch.matmul(X_w, torch.from_numpy(mu).float())
        final_X = torch.cat((X_w.view(-1, pw.shape[0]*pw.shape[1]), X_biases), dim = 1)
        return self.evaluate(final_X)
 

    def evaluate(self, X):
        outputs, hidden = self.model(X)
        return outputs
