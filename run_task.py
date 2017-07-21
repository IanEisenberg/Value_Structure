import igraph
from glob import glob
import numpy as np
import os
import subprocess
from value_struture_task import valueStructure
from utils import gen_trials, graph_from_dict


# ************Experiment Setup********************************
# subject parameters
subj = 'test'
save_dir = os.path.join('Data',subj)
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



# plot graph
g = graph_from_dict(graph)
layout = g.layout('kk')
nodes = [i['stim_index'] for i in trials]
for i,n in enumerate(nodes[0:300]):
    g.vs["color"] = ['blue' if j == n else 'red' for j in range(15)]
    igraph.plot(g, layout=layout, edge_curved=False,
                target = 'graph_%s.png' % str(i).zfill(3))

cmd = "convert -loop 0 -delay 15 *png Plots/%s.gif" % subj
process = subprocess.Popen(cmd, shell=True)
process.wait()
for f in glob('*.png'):
    os.remove(f)

