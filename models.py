from lmfit import Minimizer, Parameters
import numpy as np
# Functions to define Shift Task model (Wilson & Niv, 2012)
class BasicRLModel(object):
    """ Basic RL Model for RL task """
    def __init__(self, data, verbose=False):
        self.data = data
        # get data
        self.stims = np.unique(data.stim_indices.apply(lambda x: x[0]))
        self.vals = {i:.5 for i in self.stims}
        
        # set up class vars
        self.decay = 0
        self.beta=1
        self.eps=0
        self.lr = .01
        self.verbose=verbose
            
    def get_choice_prob(self, trial):
        stims = trial.stim_indices
        stim_values = [self.vals[stim] for stim in stims]
                       
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
        
    def update(self, trial):
        selected_stim = trial.selected_stim
        reward = trial.rewarded
        value = self.vals[selected_stim]
        delta = self.lr*(reward-value)
        self.vals[selected_stim] += delta

    def run_data(self):
        self.vals = {i:.5 for i in self.stims}
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
        params = super.get_params()
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
        
        
