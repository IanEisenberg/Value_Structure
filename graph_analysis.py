import cPickle
from glob import glob
import igraph
import numpy as np
from os import path, remove
import pandas as pd
from utils import extract_rt_relationships, get_node_colors
import subprocess

def graph_from_dict(graph_dict):
    g = igraph.Graph()
    g.add_vertices(list(set(list(graph_dict.keys()) 
                   + list([a for v in graph_dict.values() for a in v]))))
    g.add_edges([(v, a) for v in graph_dict.keys() for a in graph_dict[v]])
    return g

def graph_from_judgments(value_graph):
    from itertools import combinations
    g = igraph.Graph() 
    g.add_vertices(value_graph.keys())
    g.vs['label'] = value_graph.keys()
    for n1,n2 in combinations(value_graph.keys(),2):   
        g.add_edge(n1,n2, weight = 1-abs(value_graph[n2]-value_graph[n1]))
    return g


# load data
data_loc = path.join('Data','ProcessedData')
structuredata = pd.read_csv(path.join(data_loc, 'structuredata.csv'))
valuedata = pd.read_csv(path.join(data_loc, 'valuedata.csv'))
taskdata = cPickle.load(open(path.join(data_loc,'taskdata.pkl'),'rb'))

# visual style for graphs
visual_style = {}
colors = get_node_colors()
visual_style['vertex_color'] = colors
visual_style['vertex_size'] = 40
visual_style['bbox'] = (600,600)
visual_style['margin'] = 60
visual_style['edge_curved'] = False
visual_style['vertex_label_size'] = 25

# plot graph
graph = taskdata.values()[0]['graph']
trials = structuredata

g = graph_from_dict(graph)
layout = g.layout('kk')
nodes = [row['stim_index'] for i,row in trials.iterrows()]
# visualization stuff
g.vs['label'] = list(range(15))
# plot animations
for i,n in enumerate(nodes[0:200]):
    g.vs["color"] = ['yellow' if j == n else colors[j] for j in range(15)]
    igraph.plot(g, layout=layout, target = 'graph_%s.png' % str(i).zfill(3),
                edge_width=2, **visual_style)

cmd = "convert -loop 0 -delay 18 *png Plots/%s.gif" % trials.subjid[0]
process = subprocess.Popen(cmd, shell=True)
process.wait()
for f in glob('*.png'):
    remove(f)

# plot subject value graph
subj = 'CH'
values = taskdata[subj]['node_values'].copy()
subj_rating = valuedata.query('subjid=="%s"' % subj) \
                .groupby('stim_index').rating.mean()
values.update(subj_rating)
g = graph_from_judgments(values)
weights = [i**3*4 for i in g.es['weight']]
#value_layout = layout
value_layout = g.layout_fruchterman_reingold(weights=g.es["weight"])
igraph.plot(g, inline=False, edge_width=weights, layout=value_layout,
            **visual_style)

# plot value graph
value_graph = taskdata.values()[0]['values']
g = graph_from_judgments(value_graph)
weights = [i**3*4 for i in g.es['weight']]
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