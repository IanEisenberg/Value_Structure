import numpy as np
from utils import create_value_graph, gen_RL_trials, list_get
# test if gen_RL_trials results in E(stim_i) == value(stim_i) in the limit

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
scaling = 1
seeds = {2:.9,1:1,6:.1,7:.01}
values = create_value_graph(graph, seeds, weight=.99, steps = 3000)
values = {k:v*scaling for k,v in values.items()}
RL_trials = gen_RL_trials(stims, values, max_repeat=100)

graph_vals = {}
for i in values.keys():
    rewards = []
    for t in RL_trials:
        loc = list_get(t['stim_indices'], i, -1)
        if loc!=-1:
            rewards.append(t['rewards'][loc])
    graph_vals[i] = np.mean(rewards)
    
diff = np.mean([abs(graph_vals[key]-values[key]) for key in values.keys()])
print('True values differ from observations by %s' % diff)