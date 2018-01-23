from itertools import permutations
from math import ceil
import numpy as np
import pandas as pd
import random as r

# **********HELPER FUNCTIONS ***********************************
def create_value_graph(graph, seeds, weight = .95, steps = 1000,
                       scaling=1, offset=0):
    value_graph = {key:.5 for key in graph.keys()}
    value_graph.update(seeds)
    node = np.random.choice(value_graph.keys())
    for step in range(steps):
        connections = graph[node]
        avg_val = np.mean([value_graph[i] for i in connections])
        value_graph[node] = value_graph[node]*weight + avg_val*(1-weight)
        node = np.random.choice(graph[node])
    # scale 
    min_value = np.min(value_graph.values())
    for k in value_graph.keys():
        value_graph[k]-=min_value
    max_value = np.max(value_graph.values())
    for k in value_graph.keys():
        value_graph[k]/=max_value
    value_graph = {k:v*scaling+offset for k,v in value_graph.items()}

    # round
    for k,v in value_graph.items():
        value_graph[k] = ceil(v*1000.0)/1000.0
    return value_graph

def extract_rt_relationships(structuredata):
    def GroupColFunc(df, ind):
        return str(sorted(df.loc[ind,['stim_index','last_stim']]))
    data = structuredata.loc[:,['rt','stim_index']]
    data.loc[:,'last_stim'] = data.stim_index.shift()
    data = data[1:].query('rt!=-1')
    # group rt
    relationships = data.groupby(lambda x: GroupColFunc(data,x)).rt.median()
    N = len(structuredata.stim_index.unique())
    relational_matrix = np.zeros([N,N])
    for index, row in relationships.iteritems():
        i=int(float(index[1:4]))
        j=int(float(index[6:9]))
        relational_matrix[i,j] = relational_matrix[j,i] = 1/row
    return relational_matrix
    
def gen_sequence(graph, n_steps):
    curr_i = np.random.choice(graph.keys())
    sequence = []
    for _ in range(n_steps):
        sequence.append(curr_i)
        curr_i = np.random.choice(graph[curr_i])
    return sequence

def gen_balanced_sequence(graph, repetitions):
    sequence=list(graph.keys())*repetitions
    np.random.shuffle(sequence)
    return sequence

def gen_parsing_trials(graph, stims, n_steps=600, duration=1.5, seed=None):
    assert n_steps%15==0
    if seed:
        np.random.seed(seed)
    blocks = n_steps//15
    curr_i = np.random.choice(graph.keys())
    block_types = ['hamiltonian','random','hamiltonian_reversed','random']
    sequence = []
    for b in range(blocks):
        block_type = block_types[b%4]
        if block_type == 'hamiltonian':
            hamiltonian = [((curr_i+i)%15, block_type) for i in range(15)]
            sequence.extend(hamiltonian)
        elif block_type == 'hamiltonian_reversed':
            hamiltonian = [((curr_i-i)%15, block_type) for i in range(15)]
            sequence.extend(hamiltonian)
        else:
            for _ in range(15):
                sequence.append((curr_i, block_type))
                curr_i = np.random.choice(graph[curr_i])
    
    last_comm = None
    trials = []
    for i, (stim_i, walk_type) in enumerate(sequence[0:15]):
        comm = get_community(stim_i)
        if last_comm is not None and comm!=last_comm:
            transition=True
        else:
            transition=False
        trial = {'stim_index': stim_i,
                 'stim_file': stims[stim_i],
                 'walk_type': walk_type,
                 'community_transition': transition,
                 'duration': duration,
                 'exp_stage': 'parse'}
        last_comm = comm
        trials.append(trial)
    np.random.seed()
    return trials
        
    
    
def gen_random_RL_trials(stims, values, repeats=3,  duration=2.5, 
                         feedback_duration=1, seed=None):
    if seed:
        np.random.seed(seed)
    available_stims = values.keys()
    # get trials per condition
    trials = []
    permutes = list(permutations(available_stims,2)) * repeats
    np.random.shuffle(permutes)
    for stim1, stim2 in permutes:
        stim_values =  [values[stim1], values[stim2]]
        rewards = [int(r.random() < v) for v in stim_values]
        trial = {'stim_indices': [stim1, stim2],
                 'stim_files': [stims[stim1], stims[stim2]],
                 'rewards': rewards,
                 'values': stim_values,
                 'correct_choice': int(stim_values[1] > stim_values[0]),
                 'duration': duration,
                 'feedback_duration': feedback_duration,
                 'exp_stage': 'RL_task'}
        trials.append(trial)

    np.random.seed()
    return trials
    
def gen_structured_RL_trials(stims, values, sets=6, duration=2.5, 
                             feedback_duration=1, reward_blackout=0,
                             repeats=None, seed=None):
    """ Generates a sequence of trials fot eh RL task
    Args:
        stims: list of stim files
        values: values for each stim
        sets: number of stim sets to display
        duration: duration of stimulus
        feedback_duration: duration of feedback
        reward_blackout: number of trials after switch where reward is hidden
            If reward_blackout=0 rewards are always displayed
        seed: random seed
    """
    if seed:
        np.random.seed(seed)
    if sets == 2:
        if repeats is None:
            repeats = 6
        stim_rollout = [(1,2,6,7,11,12), 
                        (3,4,5,8,9,10,13,14)]
    elif sets == 6:
        if repeats is None:
            repeats = 20
        stim_rollout = [(1,6,11),
                        (2,7,12),
                        (3,8,13),
                        (4,5),
                        (9,10),
                        (0,14)]
    
    trials = []
    for available_stims in stim_rollout:
        # get trials per condition
        permutes = list(permutations(available_stims,2)) * repeats
        np.random.shuffle(permutes)
        for i, (stim1, stim2) in enumerate(permutes):
            # determine whether reward should be displayed
            display_reward = 1
            if i < reward_blackout:
                display_reward = 0
            # calculate reward and values
            stim_values =  (values[stim1], values[stim2])
            rewards = tuple((int(r.random() < v) for v in stim_values))
            if stim_values[1] == stim_values[0]:
                correct_choice = np.nan
            else:
                correct_choice = int(stim_values[1] > stim_values[0])
            trial = {'stim_indices': [stim1, stim2],
                     'stim_files': [stims[stim1], stims[stim2]],
                     'rewards': rewards,
                     'display_reward': display_reward,
                     'values': stim_values,
                     'correct_choice': correct_choice,
                     'duration': duration,
                     'feedback_duration': feedback_duration,
                     'stim_set': available_stims,
                     'exp_stage': 'RL_task'}
            trials.append(trial)
    np.random.seed()
    return trials
    
def gen_nbackstructure_trials(graph, stims, trial_count=100, duration=1.5, 
                         exp_stage=None, balanced=False,  seed=None,
                         n=2, prop_match=.2, allowance=.02):
    if seed:
        np.random.seed(seed)
    emp_prop_match = 0
    while (emp_prop_match < (prop_match-allowance) or 
           emp_prop_match > (prop_match+allowance)):
        sequence = gen_sequence(graph, trial_count)
        nback_match = (np.roll(sequence, n)==np.array(sequence))[n:]
        emp_prop_match = np.mean(nback_match)
    nback_match = np.append([False]*2, nback_match).astype(int)
    trials = []
    for i, stim_i in enumerate(sequence):
        trial = {'stim_index': stim_i,
                 'stim_file': stims[stim_i],
                 'duration': duration,
                 'exp_stage': exp_stage,
                 'nback_match': nback_match[i]}
        trials.append(trial)
    np.random.seed()
    return trials    
    
def gen_rotstructure_trials(graph, stims, trial_count=100, duration=1.5, 
                         exp_stage=None, balanced=False,  seed=None,
                         proportion_rotated=.15):
    if seed:
        np.random.seed(seed)
    if not balanced:
        sequence = gen_sequence(graph, trial_count)
    else:
        sequence = gen_balanced_sequence(graph, trial_count//len(graph.keys()))
    stim_counts = pd.value_counts(sequence)
    stim_rotations = {k: [0]*v for k,v in stim_counts.iteritems()}
    for k,v in stim_rotations.items():
        n_rotations = int(round(len(v)*proportion_rotated))
        v[0:n_rotations] = [90]*n_rotations
        np.random.shuffle(v)
        
    trials = []
    for stim_i in sequence:
        trial = {'stim_index': stim_i,
                 'stim_file': stims[stim_i],
                 'duration': duration,
                 'rotation': stim_rotations[stim_i].pop(),
                 'exp_stage': exp_stage}
        trials.append(trial)
    np.random.seed()
    return trials

# helper functions
def get_community(indices):
    communities = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14]]
    if type(indices) == list:
        community_labels = []
        for index in indices:
            community_labels.append([i for i, comm in enumerate(communities) if index in comm][0])
        return community_labels
        
    else:
        return [i for i, comm in enumerate(communities) if indices in comm][0]
            
    
def list_get(lst, elem, default=-1):
        try:
            thing_index = lst.index(elem)
            return thing_index
        except ValueError:
            return default
            
def get_lower(mat):
    return mat[np.tril_indices_from(mat,-1)]
    
def get_node_colors(nodes=11, subset=None):
    w1 = .67; w2 = .33
    c_colors = [[.8,0,.2], [0,.8,.4], [0,.1,1]]
    if nodes == 15:
        colors = [c_colors[0]]*5 + [c_colors[1]]*5 + [c_colors[2]]*5
        colors[0] = [i*w1+y*w2 for i,y in zip(c_colors[0],c_colors[2])]
        colors[4] = [i*w1+y*w2 for i,y in zip(c_colors[0],c_colors[1])]
        colors[5] = [i*w1+y*w2 for i,y in zip(c_colors[1],c_colors[0])]
        colors[9] = [i*w1+y*w2 for i,y in zip(c_colors[1],c_colors[2])]
        colors[10] = [i*w1+y*w2 for i,y in zip(c_colors[2],c_colors[1])]
        colors[14] = [i*w1+y*w2 for i,y in zip(c_colors[2],c_colors[0])]
    elif nodes == 11:
        colors = [c_colors[0]]*5 + [c_colors[1]] + [c_colors[2]]*5
        colors[0] = [i*w1+y*w2 for i,y in zip(c_colors[0],c_colors[2])]
        colors[4] = [i*w1+y*w2 for i,y in zip(c_colors[0],c_colors[1])]
        colors[6] = [i*w1+y*w2 for i,y in zip(c_colors[2],c_colors[1])]
        colors[10] = [i*w1+y*w2 for i,y in zip(c_colors[2],c_colors[0])]
    if subset:
        colors = [c for i,c in enumerate(colors) if i in subset]
    return colors

def scale(lst):
    lst_max = np.max(lst)
    lst_min = np.min(lst)
    scaled = list((np.array(lst)-lst_min)/(lst_max-lst_min))
    return scaled
    