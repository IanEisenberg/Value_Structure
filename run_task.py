import numpy as np
import os
from tasks.RLTask import RLTask
from tasks.NBackStructureTask import NBackStructureTask
from utils.utils import create_value_graph


# ************Experiment Setup********************************
# subject parameters
print('Enter the subject ID')
subj = raw_input('subject id: ')
save_dir = os.path.join('Data')
n_structure_trials = 1400
n_familiarization_trials = 30



# set up and shuffle stims

# graph structure
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
stims = ['images/%s.png' % str(i+1) for i in graph.keys()]
np.random.shuffle(stims)

# create value graph

np.random.seed(2222)
seeds = {2:.9, 1:1, 6:.1,7:.01}
values = create_value_graph(graph, seeds, weight=.98, steps = 3000,
                            scaling=.85, offset=.05)

np.random.seed()


# set up task
structure_trial_params = {'N': 2,
                          'num_trials': n_structure_trials,
                          'num_practice_trials': 60}

structure = NBackStructureTask(expid='structure',
                          subjid=subj, 
                          save_dir=save_dir, 
                          stim_files=stims, 
                          graph=graph, 
                          trial_params=structure_trial_params,
                          fullscreen=False)

                     
RLtask = RLTask(expid='RL',
                subjid=subj,
                save_dir=save_dir,
                stim_files=stims,
                values=values,
                sequence_type='structured',
                repeats=6,
                fullscreen=False)

# start task
structure.run_task()
points = RLtask.run_task()
