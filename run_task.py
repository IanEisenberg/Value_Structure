import cPickle
import numpy as np
import os
from value_struture_task import valueStructure
from utils import gen_trials


# ************Experiment Setup********************************
# subject parameters
subj = raw_input('subject id: ')
save_dir = os.path.join('Data')
n_structure_trials = 1000
n_familiarization_trials = 20



# set up and shuffle stims
stims = ['images/%s.png' % i for i in range(1,12)]
np.random.shuffle(stims)
# graph structure
"""
# 15 node graph
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
"""
# 11 node graph
graph = {0: [1,2,3,10],
         1: [0,2,3,4],
         2: [0,1,3,4],
         3: [0,1,2,4],
         4: [1,2,3,5],
         5: [4,6],
         6: [5,7,8,9],
         7: [6,8,9,10],
         8: [6,7,9,10],
         9: [6,7,8,10],
         10: [6,8,9,0]}

# create value graph
"""
scaling = 10
np.random.seed(2222)
seeds = {2:.9,1:1,6:.1,7:.01}
values = create_value_graph(graph, seeds, weight=.99, steps = 3000)
values = {k:v*scaling for k,v in values.items()}
cPickle.dump(values, open('values.pkl','wb'))
"""

values = cPickle.load(open('values.pkl','rb'))

# set up trials
familiarization_trials = gen_trials(graph, stims, n_familiarization_trials, 
                                    duration=None,
                                    exp_stage='familiarization_test',
                                    balanced=True)

trials = gen_trials(graph, stims, n_structure_trials, 
                    exp_stage='structure_learning')

# start task
# use seeds [0,1,10,11] for 15 node graph
task = valueStructure(subj, save_dir, stims, graph, values, [0,1,6,7],
                      trials, familiarization_trials, True)
win = task.run_task()


