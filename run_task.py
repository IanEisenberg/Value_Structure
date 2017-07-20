import numpy as np
from os import path
from value_struture_task import valueStructure


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
    

# ************Experiment Setup********************************

# shuffle stims
stims = ['images/%s.png' % i for i in range(1,16)]
np.random.shuffle(stims)
# graph structure
graph = {0: [1,2,3,14],
         1: [0,2,3,4],
         2: [0,1,3,4],
         3: [0,1,2,4],
         4: [1,2,3,5],
         5: [4,6,7,8],
         6: [5,7,8,9],
         7: [5,6,8,9],
         8: [5,6,7,9],
         9: [6,7,8,10],
         10: [9,11,12,13],
         11: [10,12,13,14],
         12: [10,11,13,14],
         13: [10,11,12,14],
         14: [11,12,13,0]}

familiarization_trials = gen_trials(graph, stims, 20, duration=None,
                                    exp_stage='familiarization_test',
                                    balanced=True)
trials = gen_trials(graph, stims, 1400, exp_stage='structure_learning')
subj = 'test'
save_dir = path.join('Data',subj)
task = valueStructure(subj, save_dir, trials, familiarization_trials)
task.run_task()