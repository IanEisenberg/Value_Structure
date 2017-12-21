from collections import OrderedDict as odict
from lmfit import Minimizer, Parameters
import numpy as np
import pandas as pd

# Functions to define Shift Task model (Wilson & Niv, 2012)
class BasicRLModel(object):
    """ Basic RL Model for RL task """
    def __init__(self, data, verbose=False):
        self.data = data
        self.init_val = .5
        self.stims = np.unique(data.stim_indices.apply(lambda x: x[0]))
        self.vals = odict({i: self.init_val for i in self.stims})
        
        # set up class vars
        self.decay = 0
        self.beta=1
        self.eps=0
        self.lr = .01
        self.verbose=verbose
            
    def get_choice_prob(self, trial):
        stims = trial.stim_indices
        stim_values = [self.get_value(stim) for stim in stims]
                       
        # compute softmax decision probs
        f = lambda x: np.e**(self.beta*x)
        softmax_values = [f(v) for v in stim_values]
        normalized = [v/np.sum(softmax_values) for v in softmax_values]
        # get prob of choice
        choice_prob = normalized[int(trial.response)]
        # incorporate eps
        choice_prob = (1-self.eps)*choice_prob + (self.eps)*(1/3)
        return choice_prob
    
    def get_log_likelihood(self):
        """ Returns summed negative log likelihood """
        probs, track_vals = self.run_data()
        neg_log_likelihood = -np.sum(np.log(probs))
        return neg_log_likelihood
        
    def get_params(self):
        return {'beta': self.beta,
                'lr': self.lr,
                'eps': self.eps}
                
    def get_value(self, stim):
        return self.vals[stim]

    def update(self, trial):
        selected_stim = trial.selected_stim
        reward = trial.rewarded
        value = self.get_value(selected_stim)
        delta = self.lr*(reward-value)
        self.vals[selected_stim] += delta
    
    def reset_vals(self):
        for k in self.vals:
            self.vals[k] = self.init_val
    def run_data(self):
        self.reset_vals()
        probs = []  
        track_vals = []
        for i, trial in self.data.iterrows():
            probs.append(self.get_choice_prob(trial))
            self.update(trial)
            track_vals.append(self.vals.copy())
        return probs, track_vals
    
    def optimize(self):
        def loss(pars):
            #unpack params
            parvals = pars.valuesdict()
            self.beta = parvals['beta']
            self.lr = parvals['lr']
            self.eps = parvals['eps']
            return self.get_log_likelihood()
        
        def track_loss(params, iter, resid):
            if iter%100==0:
                print(iter, resid)
            
        params = Parameters()
        params.add('beta', value=1, min=.01, max=100)
        params.add('eps', value=0, min=0, max=1)
        params.add('lr', value=.1, min=.000001, max=1)
        
        if self.verbose==False:
            fitter = Minimizer(loss, params)
        else:
            fitter = Minimizer(loss, params, iter_cb=track_loss)
        fitter.scalar_minimize(method='Nelder-Mead', options={'xatol': 1e-3,
                                                              'maxiter': 200})

class GraphRLModel(BasicRLModel):
    """ Extends basic RL Model with value propagation one edge in graph """
    def __init__(self, data, graph, verbose=False):
        super(GraphRLModel, self).__init__(data, verbose)
        self.graph = graph
        self.graph_decay = 0 # if 0, no propagation
    
    def get_params(self):
        params = super(GraphRLModel, self).get_params()
        params['graphydecay'] = self.graph_decay
        return params
                
    def update(self, trial):
        selected_stim = trial.selected_stim
        reward = trial.rewarded
        value = self.vals[selected_stim]
        delta = self.lr*(reward-value)
        self.vals[selected_stim] += delta
        # value propagation
        for stim in self.graph[selected_stim]:
            value = self.vals[stim]
            delta = self.lr*(reward-value)*self.graph_decay
            self.vals[stim] += delta
    
    def optimize(self):
        def loss(pars):
            #unpack params
            parvals = pars.valuesdict()
            self.beta = parvals['beta']
            self.eps = parvals['eps']
            self.graph_decay = parvals['graph_decay']
            self.lr = parvals['lr']
            return self.get_log_likelihood()
        
        def track_loss(params, iter, resid):
            if iter%100==0:
                print(iter, resid)
            
        params = Parameters()
        params.add('beta', value=1, min=.01, max=100)
        params.add('eps', value=0, min=0, max=1)
        params.add('graph_decay', value=0, min=0, max=1)
        params.add('lr', value=.1, min=.000001, max=1)
        
        if self.verbose==False:
            fitter = Minimizer(loss, params)
        else:
            fitter = Minimizer(loss, params, iter_cb=track_loss)
        fitter.scalar_minimize(method='Nelder-Mead', options={'xatol': 1e-3,
                                                              'maxiter': 200})
        
        
def SR_TD(data, alpha, gamma):
    state_df = pd.DataFrame({'s': data.stim_index,
                             'sprime': data.stim_index.shift(-1)}).iloc[:-1].astype(int)
    states = state_df.s.unique()
    M = np.identity(len(states)) # define future-state occupancy matrix
    def update(s, sprime):
        base = np.zeros(M.shape[1]); base[s]=1
        delta = base+gamma*M[sprime,:]-M[s,:]
        M[s,:] += alpha*delta
    for i, trial in state_df.iterrows():
        update(trial.s, trial.sprime)
    return M

def SR_from_transition(transitions, gamma):
    """ Computes successor representation from one-step transition matrix """
    I = np.identity(transitions.shape[1])
    M=np.linalg.inv(I-gamma*transitions)
    return M

class SR_RLModel(BasicRLModel):
    """ SR-RL Model 
    
    Russek, E. M., et al (2017). Predictive representations can link model-based reinforcement learning to model-free mechanisms. 
    
    """
    def __init__(self, RLdata, StructureData, verbose=False):
        super(SR_RLModel, self).__init__(RLdata, verbose)
        self.structure = StructureData
        self.SR_lr = .05
        self.gamma = .99
        self.M = self.SR_TD()
        
    def get_M(self):
        return self.M
        
    def get_params(self):
        params = super(SR_RLModel, self).get_params()
        params['SR_lr'] = self.SR_lr
        params['gamma'] = self.gamma
        return params
    
    def get_value(self, stim):
        return self.M[stim,:].dot(self.vals.values())
        
    def optimize(self):
        def loss(pars):
            #unpack params
            parvals = pars.valuesdict()
            # base RL params
            self.beta = parvals['beta']
            self.eps = parvals['eps']
            self.lr = parvals['lr']
            # SR params
            self.gamma = parvals['gamma']
            self.SR_lr = parvals['SR_lr']
            self.M = self.SR_TD()
            return self.get_log_likelihood()
        
        def track_loss(params, iter, resid):
            if iter%100==0:
                print(iter, resid)
            
        params = Parameters()
        params.add('beta', value=1, min=.01, max=100)
        params.add('eps', value=0, min=0, max=1)
        params.add('lr', value=.1, min=.000001, max=1)
        params.add('gamma', value=1, min=0, max=1)
        params.add('SR_lr', value=0, min=0, max=1)
        
        if self.verbose==False:
            fitter = Minimizer(loss, params)
        else:
            fitter = Minimizer(loss, params, iter_cb=track_loss)
        fitter.scalar_minimize(method='Nelder-Mead', options={'xatol': 1e-3,
                                                              'maxiter': 200})
        
    def SR_TD(self):
        state_df = pd.DataFrame({'s': self.structure.stim_index,
                                 'sprime': self.structure.stim_index.shift(-1)}).iloc[:-1].astype(int)
        states = state_df.s.unique()
        M = np.identity(len(states)) # define future-state occupancy matrix
        def update(s, sprime):
            base = np.zeros(M.shape[1]); base[s]=1
            delta = base+self.gamma*M[sprime,:]-M[s,:]
            M[s,:] += self.SR_lr*delta
        for i, trial in state_df.iterrows():
            update(trial.s, trial.sprime)
        return M















