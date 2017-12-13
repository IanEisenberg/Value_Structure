from itertools import permutations
import numpy as np
import pandas as pd
import random as r

# **********HELPER FUNCTIONS ***********************************
def create_value_graph(graph, seeds, weight = .95, steps = 1000):
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

def gen_RL_trials(stims, values, duration=1.5, max_repeat=2, seed=None):
    if seed:
        np.random.seed(seed)
    stim_rollout = [[1,13], # 1> 13
                    [5,9], # 5 > 9
                    [6,8], # 6 > 8
                    [2,7,12], # 2 > 7 > 12
                    [0,3,11,14], # 0 > 3 > 11 > 14
                    [4,10] # 4 > 10
                    ]
                    
    all_trials = []
    while len(stim_rollout) > 0:
        # get trials per condition
        trials = []
        available_stims = stim_rollout.pop(0)
        permutes = list(permutations(available_stims,2)) * max_repeat
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
                     'stim_set': available_stims}
            trials.append(trial)
        all_trials.append(trials)
    # add final section with all stims
    trials = []
    available_stims = values.keys()
    permutes = list(permutations(available_stims,2))
    np.random.shuffle(permutes)
    for stim1, stim2 in permutes:
        rewards = [r.random() < v for v in [values[stim1], values[stim2]]]
        trial = {'stim_indices': [stim1, stim2],
                 'stim_files': [stims[stim1], stims[stim2]],
                 'rewards': rewards,
                 'duration': duration,
                 'stim_set': available_stims}
        trials.append(trial)
    all_trials.append(trials)
    
    np.random.seed()
    return all_trials
    

    
    
def gen_structure_trials(graph, stims, trial_count=100, duration=1.5, 
                         exp_stage=None, balanced=False, seed=None):
    if seed:
        np.random.seed(seed)
    if not balanced:
        sequence = gen_sequence(graph, trial_count)
    else:
        sequence = gen_balanced_sequence(graph, 2)
    stim_counts = pd.value_counts(sequence)
    stim_rotations = {k: [0]*v for k,v in stim_counts.iteritems()}
    for k,v in stim_rotations.items():
        n_rotations = int(round(len(v)*.2))
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
    