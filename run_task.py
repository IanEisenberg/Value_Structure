import numpy as np
import os
from tasks.RLTask import RLTask
from tasks.StructureTask import StructureTask
from utils.utils import create_value_graph, gen_structure_trials


# ************Experiment Setup********************************
# subject parameters
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


# set up trials
familiarization_trials = gen_structure_trials(graph, 
                                              stims, 
                                              n_familiarization_trials, 
                                              duration=None,
                                              exp_stage='familiarization_test',
                                              balanced=True)

structure_trials = gen_structure_trials(graph, 
                                        stims, 
                                        n_structure_trials, 
                                        exp_stage='structure_learning',
                                        proportion_rotated=.2)




# start task
# use seeds [0,1,10,11] for 15 node graph
structure = StructureTask(expid='structure',
                          subjid=subj, 
                          save_dir=save_dir, 
                          stim_files=stims, 
                          graph=graph, 
                          trials=structure_trials, 
                          familiarization_trials=familiarization_trials, 
                          fullscreen=True)

                     
RLtask = RLTask(expid='RL',
                subjid=subj,
                save_dir=save_dir,
                stim_files=stims,
                values=values,
                sequence_type='semistructured',
                repeats=6,
                fullscreen=True)

structure.run_task()
points = RLtask.run_task()
