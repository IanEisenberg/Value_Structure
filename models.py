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
        self.beta=1
        self.eps=0
        self.lr = .01
        self.verbose=verbose
    
    def _calc_choice_probs(self, trial):
        stims = trial['stim_indices']
        stim_values = [self._get_value(stim) for stim in stims]
                       
        # compute softmax decision probs
        f = lambda x: np.e**(self.beta*x)
        softmax_values = [f(v) for v in stim_values]
        normalized = [v/np.sum(softmax_values) for v in softmax_values]
        return normalized
    
    def _get_choice_prob(self, trial):
        probs = self._calc_choice_probs(trial)
        # get prob of choice
        choice_prob = probs[int(trial['response'])]
        # incorporate eps
        choice_prob = (1-self.eps)*choice_prob + (self.eps)*(1/3)
        return choice_prob
    
    def _get_value(self, stim):
        return self.vals[stim]
    
    def _reset_vals(self):
        for k in self.vals:
            self.vals[k] = self.init_val
            
    def _update(self, trial):
        selected_stim = trial['selected_stim']
        reward = trial['rewarded']
        value = self._get_value(int(selected_stim))
        delta = self.lr*(reward-value)
        self.vals[selected_stim] += delta
    
    def _wipe_data(self, data):
        cols = ['correct', 'response','rewarded', 'rt', 
                'selected_stim', 'selected_value']
        for col in cols:
            if col in data.columns:
                data.loc[:, col] = np.nan
        
    def get_log_likelihood(self, start=None, stop=None):
        """ Returns summed negative log likelihood """
        probs, track_vals = self.run_data()
        if start:
            probs = probs[start:]
        if stop:
            probs = probs[:stop]
        neg_log_likelihood = -np.sum(np.log(probs))
        return neg_log_likelihood
        
    def get_params(self):
        return {'beta': self.beta,
                'lr': self.lr,
                'eps': self.eps}
                
    def update_params(self, params):
        self.__dict__.update(params)
        
    def run_data(self):
        self._reset_vals()
        probs = []  
        track_vals = []
        for i, trial in self.data.iterrows():
            probs.append(self._get_choice_prob(trial))
            self._update(trial)
            track_vals.append(self.vals.copy())
        return probs, track_vals
    
    def simulate(self, prob_match=True, data=None):
        if data is None:
            data = self.data
        self._reset_vals()
        sim_data = []
        for i, trial in data.iterrows():
            trial = trial.to_dict()
            # make decision
            probs = self._calc_choice_probs(trial)
            if prob_match:
                choice = np.random.choice([0,1], p=probs)
            else:
                choice = np.argmax(probs)
            trial['response_prob'] = probs[choice]
            trial['response'] = choice
            trial['rewarded'] = trial['rewards'][choice]
            trial['selected_stim']  = trial['stim_indices'][choice]
            trial['selected_value'] = trial['values'][choice]
            if np.isnan(trial['correct_choice']) == False:
                trial['correct']  = choice==trial['correct_choice']
            else:
                trial['correct'] = np.nan
            sim_data.append(trial)
            # update
            if trial['display_reward']:
                self._update(trial)
        # change type of columns
        sim_data = pd.DataFrame(sim_data)
        sim_data.selected_stim = sim_data.selected_stim.astype(int)
        sim_data.correct = sim_data.correct.astype(float)
        return sim_data
            
    def optimize(self, start=None, stop=None):
        def loss(pars):
            #unpack params
            parvals = pars.valuesdict()
            self.beta = parvals['beta']
            self.lr = parvals['lr']
            self.eps = parvals['eps']
            return self.get_log_likelihood(start, stop)
        
        def track_loss(params, iter, resid):
            if iter%100==0:
                print(iter, resid, params)
            
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
        
    def optimize_acc(self, data=None):
        def loss(pars):
            #unpack params
            parvals = pars.valuesdict()
            self.beta = parvals['beta']
            self.lr = parvals['lr']
            self.eps = parvals['eps']
            sim_data = self.simulate(data=data, prob_match=True)
            # get probability of correct response
            prob_correct = abs((1-sim_data.correct)-sim_data.response_prob)
            return -np.sum(np.log(prob_correct))
        
        def track_loss(params, iter, resid):
            if iter%100==0:
                print(iter, resid, params)
            
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
    
    def _get_value(self, stim):
        return self.M[stim,:].dot(self.vals.values())/np.sum(self.M[stim,:])
    
    def get_M(self):
        return self.M
        
    def get_params(self):
        params = super(SR_RLModel, self).get_params()
        params['SR_lr'] = self.SR_lr
        params['gamma'] = self.gamma
        return params
        
    def optimize(self, start=None, stop=None):
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
            return self.get_log_likelihood(start, stop)
        
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
        
    def optimize_acc(self, data=None):
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
            sim_data = self.simulate(data=data, prob_match=True)
            prob_correct = abs((1-sim_data.correct)-sim_data.response_prob)
            return -np.sum(np.log(prob_correct))
        
        def track_loss(params, iter, resid):
            if iter%100==0:
                print(iter, resid)
            
        params = Parameters()
        params.add('beta', value=1, min=.01, max=100)
        params.add('eps', value=0, min=0, max=1)
        params.add('lr', value=.1, min=.000001, max=1)
        params.add('gamma', value=1, min=0, max=1)
        params.add('SR_lr', value=.05, min=0, max=1)
        
        if self.verbose==False:
            fitter = Minimizer(loss, params)
        else:
            fitter = Minimizer(loss, params, iter_cb=track_loss)
        fitter.scalar_minimize(method='Nelder-Mead', options={'xatol': 1e-3,
                                                              'maxiter': 200})
        
    
    def simulate(self, prob_match=True, data=None):
        self.M = self.SR_TD()
        sim_data = super(SR_RLModel, self).simulate(prob_match, data)
        return sim_data
        
    def SR_TD(self):
        state_df = pd.DataFrame({'s': self.structure.stim_index,
                                 'sprime': self.structure.stim_index.shift(-1)}).iloc[:-1].astype(int)
        states = state_df.s.unique()
        M = np.identity(len(states)) # define future-state occupancy matrix
        def _update(s, sprime):
            base = np.zeros(M.shape[1]); base[s]=1
            delta = base+self.gamma*M[sprime,:]-M[s,:]
            M[s,:] += self.SR_lr*delta
        for i, trial in state_df.iterrows():
            _update(trial['s'], trial['sprime'])
        return M

class GraphRLModel(BasicRLModel):
    """ Extends basic RL Model with value propagation one edge in graph """
    def __init__(self, data, graph, verbose=False):
        super(GraphRLModel, self).__init__(data, verbose)
        self.graph = graph
        self.graph_decay = 0 # if 0, no propagation
    
    def _update(self, trial):
        selected_stim = trial['selected_stim']
        reward = trial['rewarded']
        value = self.vals[selected_stim]
        delta = self.lr*(reward-value)
        self.vals[selected_stim] += delta
        # value propagation
        for stim in self.graph[selected_stim]:
            value = self.vals[stim]
            delta = self.lr*(reward-value)*self.graph_decay
            self.vals[stim] += delta
            
    def get_params(self):
        params = super(GraphRLModel, self).get_params()
        params['graphydecay'] = self.graph_decay
        return params
                
    def optimize(self, start=None, stop=None):
        def loss(pars):
            #unpack params
            parvals = pars.valuesdict()
            self.beta = parvals['beta']
            self.eps = parvals['eps']
            self.graph_decay = parvals['graph_decay']
            self.lr = parvals['lr']
            return self.get_log_likelihood(start, stop)
        
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


