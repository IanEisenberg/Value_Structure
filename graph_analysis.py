from glob import glob
import igraph
import os
import subprocess

def graph_from_dict(graph_dict):
    g = igraph.Graph()
    g.add_vertices(list(set(list(graph_dict.keys()) 
                   + list([a for v in graph_dict.values() for a in v]))))
    g.add_edges([(v, a) for v in graph_dict.keys() for a in graph_dict[v]])
    return g

# load data

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

