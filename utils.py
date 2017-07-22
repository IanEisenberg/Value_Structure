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
                 'rotation': 90*np.random.choice([0,1],p=[.85,.15]),
                 'exp_stage': exp_stage}
        trials.append(trial)
    return trials

