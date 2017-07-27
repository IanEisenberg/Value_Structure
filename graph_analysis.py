import cPickle
from glob import glob
import igraph
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
from os import path, remove
import pandas as pd
from utils import extract_rt_relationships, get_node_colors
from graph_utils import adj_from_judgments, graph_from_value_adj, average_adjs
from graph_utils import graph_from_dict
import seaborn as sns
import subprocess


# load data
data_loc = path.join('Data','ProcessedData')
structuredata = pd.read_csv(path.join(data_loc, 'structuredata.csv'))
valuedata = pd.read_csv(path.join(data_loc, 'valuedata.csv'))
taskdata = cPickle.load(open(path.join(data_loc,'taskdata.pkl'),'rb'))

# visual style for graphs
visual_style = {}
colors = get_node_colors()
visual_style['vertex_size'] = 40
visual_style['bbox'] = (600,600)
visual_style['margin'] = 60
visual_style['edge_curved'] = False
visual_style['vertex_label_size'] = 25

#********************************************
# Plot task structure
#********************************************
# plot graph
graph = taskdata.values()[0]['graph']
value_graph = taskdata.values()[0]['node_values']
trials = structuredata

g = graph_from_dict(graph)
layout = g.layout('kk')
nodes = [row['stim_index'] for i,row in trials.iterrows()]
n_nodes = np.max(nodes)+1
# visualization stuff
g.vs['label'] = list(range(n_nodes))
# plot animations
for i,n in enumerate(nodes[0:200]):
    g.vs["color"] = ['yellow' if j == n else colors[j] for j in range(n_nodes)]
    igraph.plot(g, layout=layout, target = 'graph_%s.png' % str(i).zfill(3),
                edge_width=2, **visual_style)

cmd = "convert -loop 0 -delay 18 *png Plots/temporal_graph.gif" % trials.subjid[0]
process = subprocess.Popen(cmd, shell=True)
process.wait()
for f in glob('*.png'):
    remove(f)

# plot value graph
value_adj = adj_from_judgments(value_graph)
g = graph_from_value_adj(value_adj)
# plot heatmap of graph
f=sns.plt.figure(figsize = (12,8))
sns.heatmap(value_adj, square=True)
sns.plt.title('Value Differences Between Nodes', fontsize = 24)
f.savefig('Plots/value_heatmap.png')
# plot graph
g.vs['color'] = colors
weights = [i for i in g.es['weight']]
#value_layout = layout
value_layout = g.layout_fruchterman_reingold(weights=weights)
igraph.plot(g, inline=False, edge_width=weights, layout=value_layout,
            target = 'Plots/value_graph.png',
            **visual_style)

value_adjs = {}
for subj in taskdata.keys():
    # plot subject value graph
    subj_rating = dict(valuedata.query('subjid=="%s"' % subj) \
                    .groupby('stim_index').reg_rating.mean())
    value_adj = adj_from_judgments(subj_rating)
    if len(np.unique(value_adj)) != 1:
        value_adjs[subj] = value_adj

avg_value_adj = average_adjs(value_adjs)

# plot individual heatmaps
nrows = int(ceil(len(value_adjs)/2.0))
f, ax = plt.subplots(nrows=nrows, ncols=2,
                     figsize=(10, nrows*5))
i=0
for subj,adj in value_adjs.items():
    sns.heatmap(adj, ax=f.axes[i])
    f.axes[i].set_title(subj, fontsize=20)
    i+=1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Individual Value Distances', fontsize=20)

# plot average heatmap of graph
sns.heatmap(avg_value_adj)
g = graph_from_value_adj(value_adj)

# plot graph
colors = get_node_colors(subset = g.vs['label'])
g.vs['color'] = colors
weights = [w**3*4 for w in g.es['weight']]
#value_layout = layout
value_layout = g.layout_fruchterman_reingold(weights=g.es["weight"])
igraph.plot(g, inline=False, edge_width=weights, layout=value_layout,
            **visual_style)

















# plot RT graph
relational_matrix = extract_rt_relationships(trials)
g = igraph.Graph.Weighted_Adjacency(relational_matrix.tolist(), mode='undirected')
weights = np.array([i for i in g.es['weight']])
weights = (weights-weights.mean())/weights.std()+3

igraph.plot(g, inline=False, edge_width=weights,
            **visual_style)