import cPickle
from glob import glob
import igraph
from os import path, remove
import pandas as pd
from utils import get_node_colors
import subprocess

def graph_from_dict(graph_dict):
    g = igraph.Graph()
    g.add_vertices(list(set(list(graph_dict.keys()) 
                   + list([a for v in graph_dict.values() for a in v]))))
    g.add_edges([(v, a) for v in graph_dict.keys() for a in graph_dict[v]])
    return g

# load data
data_loc = path.join('Data','ProcessedData')
structuredata = pd.read_csv(path.join(data_loc, 'structuredata.csv'))
taskdata = cPickle.load(open(path.join(data_loc,'taskdata.pkl'),'rb'))


# plot graph
graph = taskdata.values()[0]['graph']
trials = structuredata

g = graph_from_dict(graph)
layout = g.layout('kk')
nodes = [row['stim_index'] for i,row in trials.iterrows()]
# visualization stuff
g.vs['label'] = list(range(15))
colors = get_node_colors()
# plot animations
for i,n in enumerate(nodes[0:200]):
    g.vs["color"] = ['yellow' if j == n else colors[j] for j in range(15)]
    igraph.plot(g, layout=layout, edge_curved=False,
                target = 'graph_%s.png' % str(i).zfill(3),
                edge_width=2, vertex_size=50, vertex_label_size=25,
                bbox = (600, 600),
                margin=60)

cmd = "convert -loop 0 -delay 18 *png Plots/%s.gif" % trials.subjid[0]
process = subprocess.Popen(cmd, shell=True)
process.wait()
for f in glob('*.png'):
    remove(f)

