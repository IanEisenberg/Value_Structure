import numpy as np
from os import path
from value_struture_task import valueStructure
from utils import gen_trials, graph_from_dict


# ************Experiment Setup********************************
# subject parameters
subj = 'test'
save_dir = path.join('Data',subj)
n_structure_trials = 1400
n_familiarization_trials = 30



# set up and shuffle stims
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

# plot graph
g = graph_from_dict(graph)


# set up trials
familiarization_trials = gen_trials(graph, stims, n_familiarization_trials, 
                                    duration=None,
                                    exp_stage='familiarization_test',
                                    balanced=True)

trials = gen_trials(graph, stims, n_structure_trials, 
                    exp_stage='structure_learning')

# start task
task = valueStructure(subj, save_dir, stims, graph,
                      trials, familiarization_trials, False)
task.run_task()