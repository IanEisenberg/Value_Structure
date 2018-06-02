import numpy as np
import os
from tasks.RLTask import RLTask
from tasks.NBackStructureTask import NBackStructureTask
from tasks.ParsingTask import ParsingTask
from tasks.RotationStructureTask import RotationStructureTask
from utils.utils import create_value_graph


# ************Experiment Setup********************************
# subject parameters
print('Enter the subject ID')
subj = input('subject id: ')
save_dir = os.path.join('Data')
structure_task = 'nback'
n_structure_trials = 1400
n_structure_practice_trials = 60
n_parse_trials = 600
fullscreen=False


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
seeds = {1:0, 12:1}
values = create_value_graph(graph, seeds, weight=.97, steps = 3000,
                            scaling=.8, offset=.1)
# hardwire the connector nodes to all have the same value
for k in [0,14,4,5,9,10]:
    values[k] = .5
np.random.seed()

# Set up Structure Task
if structure_task == 'rotation':
    # set up task
    structure_trial_params = {'num_trials': n_structure_trials,
                              'proportion_rotated': .15,
                              'seed': 10101}
    
    structure = RotationStructureTask(expid='structure',
                              subjid=subj, 
                              save_dir=save_dir, 
                              stim_files=stims, 
                              graph=graph, 
                              trial_params=structure_trial_params,
                              fullscreen=fullscreen)
elif structure_task == 'nback':
    # set up task
    structure_trial_params = {'N': 2,
                              'num_trials': n_structure_trials,
                              'num_practice_trials': n_structure_practice_trials,
                              'seed': 10101}
    
    structure = NBackStructureTask(expid='structure',
                              subjid=subj, 
                              save_dir=save_dir, 
                              stim_files=stims, 
                              graph=graph, 
                              trial_params=structure_trial_params,
                              fullscreen=fullscreen)

# Set up RL task
RL_trial_params = {'sets': 6,
                   'reward_blackout': 10}
RLtask = RLTask(expid='RL',
                subjid=subj,
                save_dir=save_dir,
                stim_files=stims,
                values=values,
                sequence_type='structured',
                fullscreen=fullscreen,
                trial_params=RL_trial_params)

# Set up parse task
parse_trial_params = {'num_trials': n_parse_trials,
                     'seed': 10101}
parse = ParsingTask(expid='parse',
                    subjid=subj,
                    save_dir=save_dir,
                    stim_files=stims,
                    graph=graph,
                    fullscreen=fullscreen,
                    trial_params=parse_trial_params)

# start task
structure.run_task()
points = RLtask.run_task()
parse = parse.run_task()
