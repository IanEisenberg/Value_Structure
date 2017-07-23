import numpy as np

# **********HELPER FUNCTIONS ***********************************
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