import numpy as np

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
    
def gen_trials(graph, stims, trial_count=100, duration=1.5, exp_stage=None,
               balanced=False):
    if not balanced:
        sequence = gen_sequence(graph, trial_count)
    else:
        sequence = gen_balanced_sequence(graph, 2)
        
    trials = []
    for stim_i in sequence:
        trial = {'stim_index': stim_i,
                 'stim_file': stims[stim_i],
                 'duration': duration,
                 'rotation': 90*np.random.choice([0,1],p=[.8,.2]),
                 'exp_stage': exp_stage}
        trials.append(trial)
    return trials

def get_node_colors():
    c_colors = [[.8,0,.2], [0,.8,.4], [0,.1,1]]
    colors = [c_colors[0]]*5 + [c_colors[1]]*5 + [c_colors[2]]*5
    colors[0] = [i*.7+y*.3 for i,y in zip(c_colors[0],c_colors[2])]
    colors[4] = [i*.7+y*.3 for i,y in zip(c_colors[0],c_colors[1])]
    colors[5] = [i*.7+y*.3 for i,y in zip(c_colors[1],c_colors[0])]
    colors[9] = [i*.7+y*.3 for i,y in zip(c_colors[1],c_colors[2])]
    colors[10] = [i*.7+y*.3 for i,y in zip(c_colors[2],c_colors[1])]
    colors[14] = [i*.7+y*.3 for i,y in zip(c_colors[2],c_colors[0])]
    return colors