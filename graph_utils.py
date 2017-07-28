import igraph
import numpy as np
import pandas as pd

def graph_from_dict(graph_dict):
    g = igraph.Graph()
    g.add_vertices(list(set(list(graph_dict.keys()) 
                   + list([a for v in graph_dict.values() for a in v]))))
    g.add_edges([(v, a) for v in graph_dict.keys() for a in graph_dict[v]])
    return g

def adj_from_judgments(value_graph):
    from itertools import combinations
    value_graph = value_graph.copy()
    # normalize values
    values = value_graph.values()
    vmax = np.max(values); vmin = np.min(values)
    if abs(vmin - vmax) > 1E-5:
        value_graph = {k:(v-vmin)/(vmax-vmin) for k,v in value_graph.items()}
    else:
        value_graph = {k:.5 for k in value_graph.keys()}
    adj = np.ones([len(values), len(values)])
    nodes = value_graph.keys()
    for n1,n2 in combinations(nodes,2):   
        i = nodes.index(n1)
        j = nodes.index(n2)
        adj[i,j] = adj[j,i] = 1-(abs(value_graph[n2]-value_graph[n1]))
    adj = pd.DataFrame(adj, index=value_graph.keys(), columns=value_graph.keys())
    return adj

def average_adjs(adjs, weights=None):
    subjs, adjs = zip(*adjs.items())
    adj_matrices = [i.as_matrix() for i in adjs]
    if weights:
        weights = [weights[s] for s in subjs]        
        adj_matrices = [adj*weights[i] for i,adj in enumerate(adj_matrices)]
    avg = np.mean(adj_matrices, axis=0)
    avg = pd.DataFrame(avg, index=adjs[0].index, columns=adjs[0].columns)
    return avg

def graph_from_value_adj(value_adj):
    value_adj = value_adj.copy()
    diag_i = zip(*np.diag_indices_from(value_adj))
    for x,y in diag_i:
        value_adj.iloc[x,y] = 0
    g=igraph.Graph.Weighted_Adjacency(value_adj.as_matrix().tolist(), 
                                      mode='undirected')
    g.vs['label'] = value_adj.columns
    return g

def graph_to_dataframe(G):
    matrix = graph_to_matrix(G)
    graph_dataframe = pd.DataFrame(data = matrix, columns = G.vs['label'], index = G.vs['label'])
    return graph_dataframe

def graph_to_matrix(G):
    graph_mat = np.array(G.get_adjacency(attribute = 'weight').data)
    graph_mat[np.diag_indices_from(graph_mat)]=1
    return graph_mat