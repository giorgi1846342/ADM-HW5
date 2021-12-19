#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import gzip
import pandas as pd
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import json
from beautifultable import BeautifulTable
import networkx as nx
import matplotlib.pyplot as plt
import collections
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from collections import deque
from sys import maxsize as INT_MAX
import itertools
from joblib import Parallel, delayed
from matplotlib.patches import Polygon
import scipy as sp
import scipy.sparse

# PRE-PROCESSING
def create_graph_1(df): # "parent" node, child node
    g={} #empty
    for i, row in tqdm(df.iterrows(), total=df.shape[0]): #Read the dataset by rows
        ym = int(str(row[2])[:-2]) #year_month, Key1
        tuple_line = [int(row[0]), int(str(row[2])[-2:])] #tuple: node u, weight
        key_row = int(row[1])
        if ym in g: #if Key1 already exists
            if key_row in g[ym]: #if Key2 already exists (node ​​v)
                   g[ym][key_row].append(tuple_line) #append tuple
            else:
                   g[ym][key_row] = [tuple_line] #append Key 2
        else:
            g[ym] = {key_row : [tuple_line]} #if there was "nothing" append everything
    return g

def graph_merge_f(diz, json):
    for year in tqdm(json):
        for d in json[year]:
            if year in diz:
                if d in diz[year]:
                    diz[year][d].extend(json[year][d])
                else:
                    diz[year][d] = json[year][d]
            else:
                diz[year] = {}
                diz[year][d] = json[year][d]
    return diz

'''
Function to use for functionalities 2,3 and 4: you give it a time interval (two dates in the "year-month" format) and it returns you the graph / dictionary only in that interval.
You need to pass this function at the beginning of yput functionality funct. For the visualization,maybe you can convert the graph that you obtein with 
networkx.convert.from_dict_of_lists. But it can't read our structure very well, probably is better to use other libraries for plotting in general (?).
'''
def sub_graph(first_date, second_date, json):
    start = int(first_date.replace("-",""))
    end = int(second_date.replace("-",""))
    result = dict()
    for year in json:
        if start <= int(year) <= end:
            node_year = json[year] # insieme di nodi di quell'anno
            for node in node_year:
                if node not in result:
                    result[node] = [json[year][node]]
                else:
                    result[node].append(json[year][node])
    return result

# GENERAL (for functionalities 2 and 3)
def dictionaries(subgraph, weighted):
    '''
    input: subgraph from sub_graph function
    output: dictionaries edges, directed; nodes: set of all nodes in subgraph
    '''
    # dictionary edges
    # _key_: tuple (source, target); _value_: edge occurrence (could be the weight)
    edges = defaultdict()
    # dictionary directed
    # _key_: source; _value_: all possible targets (adjacent nodes) 
    directed = defaultdict()
    # nodes: set of all nodes in the sub-graph
    nodes = set()
    # build dictionaries
    for source in subgraph.keys():
        nodes.add(int(source))
        aux = list(itertools.chain(*subgraph[source]))
        directed[int(source)] = set()
        for target in aux:
            nodes.add(target[0])
            directed[int(source)].add(target[0])
            if (int(source), target[0]) not in edges.keys():
                edges[(int(source), target[0])] = 1
            else:
                edges[(int(source), target[0])] += 1
        directed[int(source)] = list(directed[int(source)])
    
    # weigh the edges according to occurrences
    for edge in edges.keys():
        if weighted==True:
            edges[edge] = round(edges[edge]**(-1/5),3)
        else:
            edges[edge] = 1
    
    return edges, directed, nodes

# FUNCTIONALITY 1
def functionality_1(num_graph):
    if num_graph == 1 :
        with open('/content/drive/MyDrive/ADM-HW5/graph_a2q.json') as json_file:
            g = json.load(json_file)
    elif num_graph == 2:
        with open('/content/drive/MyDrive/ADM-HW5/graph_c2q.json') as json_file:
            g = json.load(json_file)
    elif num_graph == 3:
        with open('/content/drive/MyDrive/ADM-HW5/graph_c2a.json') as json_file:
            g = json.load(json_file)
    return extract_feature(g)


def extract_feature(g):
    d = "True"
    #Directed or not
    #number of nodes and nodes
    user = set()
    number_edge = 0
    occurences = dict()
    for year in g:
        for node in g[year]:
            occurences[node] = occurences.get(node, 0) + 1
            user.add(node)
            for v in g[year][node]:
                occurences[v[0]] = occurences.get(v[0], 0) + 1  
                user.add(v[0])
                number_edge += 1
    number_user = len(user)

    #Average link for users
    average = round(number_edge/ number_user,2)
    #Density degree
    density = round(number_edge / (number_user * (number_user - 1)),3)
    #Sparse or dense  
    type_graph = ""    
    if density >= 0.5:
        type_graph = "DENSE"
    else:
        type_graph = "SPARSE"

    table = BeautifulTable()
    table.append_row(['IS DIRECTED?',d])
    table.append_row(['Number of user :',number_user])
    table.append_row(['Number of answer/comment :',number_edge])
    table.append_row(['Average number of links per user :',average])
    table.append_row(['Density  :',density])
    table.append_row(['The graph is :',type_graph])
    print(table)

    degrees = occurences.values()
    occurences = Counter(degrees)
    sorted_dict = collections.OrderedDict(sorted(occurences.items()))
    a = {k: v / number_user for k, v in sorted_dict.items()}
    lists = a.items() # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples

    print("The degree with the highest probability is:",x[int(max(y))])
    print("With probability:",round(max(y),2))


    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(121)
    ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
    ax1.plot(x, y)
    ax1.set_title('Degree distr.')
    ax1.set_xlabel('degree k')
    ax1.set_ylabel('P(k) = nk / n')

    ax2 = plt.subplot(122)

    ax2.plot(x, y)
    ax2.set_title(' Degree distr. Zoomed in')
    ax2.set_xlabel('degree k')
    ax2.set_ylabel('P(k) = nk / n')
    ax2.set_ylim(top=0.00001,bottom=0)
    ax2.set_xlim(0,4000)

    plt.show()


# FUNCTIONALITY 3
def find_paths(source, edges, directed, nodes):
    '''
    dijkstra implementation
    input: source node, output function dictionaries(check above)
    output: dictionaries distances, paths, route
    '''
    n = len(nodes)
    
    # key: node
    distances = defaultdict() # value: shortest distance from source node
    paths = defaultdict() # value: number shortest paths from source node
    visited = defaultdict() # value: boolean
    route = defaultdict() # value: "parent" node closest to the source node
    
    # inizialize dicts
    for node in nodes:
        distances[node] = INT_MAX 
        paths[node] = 0
        visited[node] = False
    
    distances[source] = 0
    paths[source] = 1
    route[source] = 0
 
    q = deque()
    q.append(source)
    visited[source] = True
    
    
    while q:
        current = q[0]
        q.popleft()
        
        # nodes without out-edges
        if current not in directed.keys():
            # add unvisited node to the queue
            if not visited[current]:
                q.append(current)
                visited[current] = True
        
        else:
            # for all neighboring nodes of current one
            for x in directed[current]: 
                    
                # add unvisited node to the queue
                if not visited[x]:
                    q.append(x)
                    visited[x] = True
                    route[x] = current

                # check if there is a better path
                if distances[x] > distances[current] + edges[(current, x)]: # +1 unweighted
                    distances[x] = distances[current] + edges[(current, x)] # +1 unweighted
                    paths[x] = paths[current]

                # additional shortest paths found
                elif distances[x] == distances[current] + edges[(current, x)]: # +1 unweightes
                    paths[x] += paths[current]
           
    return distances, paths, route


def sub_path(source, target, route):
    path = [target]
    
    while target != source:
        target = route[target]
        path.append(target)
        
    path.reverse()    
    return path


def shortest_walk(node_list, edges, directed, nodes):
    walk = []
    distance = []
    for i in range(len(node_list) - 1):
        distances, _, route = find_paths(node_list[i], edges, directed, nodes)
        if  distances[node_list[i+1]] < 10000:
            path = sub_path(node_list[i], node_list[i+1], route)
            distance.append(distances[node_list[i+1]])
            walk.append(path)
        else:
            print("Walk not possible!")
            break
    return walk, distance


#FUNCTIONALITY 2
def functionality_2(source, merge, start, end, metric, visualize):
    start_period = f"{start[0]}-0{start[1]}"
    span = f"{start[0]}-0{start[1]}"
    end_period = f"{end[0]}-0{end[1]}"
    
    metric_story = []
    period_story = []
    
    while span != end_period:
        sgraph = sub_graph(start_period, span, merge)
        edges, directed, nodes = dictionaries(sgraph, weighted=False)
        
        if metric == "degree":
            print(f'Span: from {start_period} to {span}')
            period_story.append((f'to {span}'))
            metric_1 = degree_centrality(source, edges, nodes)
            metric_story.append(metric_1)
            span = f"{start[0]}-0{start[1]+1}"
            start[1] = start[1]+1
            print('')
            
        elif metric == "closeness":
            #include weighted version
            print(f'Span: from {start_period} to {span}')
            period_story.append((f'to {span}'))
            _, metric_2, _ = closeness_centrality(source, directed, nodes, stop_level=1000)
            metric_story.append(metric_2)
            span = f"{start[0]}-0{start[1]+1}"
            start[1] = start[1]+1
            print('')
            
    if visualize==True:
        sgraph = sub_graph(start_period, start_period, merge)
        edges, directed, nodes = dictionaries(sgraph, weighted=False)
        print(f'Source node {source} and its neighbours in {start_period}')
        visualize_graph(source, directed, edges)
    
    plt.figure(figsize=(8,4))
    plt.plot(period_story, metric_story, 'ro')
    plt.ylabel(f'{metric} centrality of node {source}')
    plt.xlabel(f'from {start_period}')


def flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):
                l.append(elt2)
        else:
            l.append(elt)
    return l


def degree_centrality(source, edges, nodes):
    edges_arr = np.array(list(edges.keys()))
    # number of edges that go out of reference node
    out_degree = edges_arr[edges_arr[:,0]==source].shape[0]
    # out_degree = len([a for a in list(edges.keys()) if a[0] == source])
    # number of edges that go into reference node
    in_degree = edges_arr[edges_arr[:,1]==source].shape[0]
    # in_degree = len([a for a in list(edges.keys()) if a[1] == source])
    # degree centrality: normalized number of neighboring nodes
    deg_centr = (in_degree+out_degree) / (len(nodes)-1)
    print(f"The subgraph consists of {len(nodes)} nodes and {len(edges.keys())} edges.")
    print(f"The reference node {source} has {in_degree+out_degree} neighboring nodes ({out_degree} out; {in_degree} in) and its degree centrality is equal to {round(deg_centr, 8)}")
    return round(deg_centr, 8)


def closeness_centrality(source, directed, nodes, stop_level):
    # optimized function for directed graph with unitary edges
    # dictionary steps: each level contains neighboring nodes (never visited!) of the previous one
    steps = defaultdict()
    visited = set()
    visited.add(source)
    steps[1] = set(directed[source])
    i = 1
    while steps[i] != set() or visited == set(nodes):  
        steps[i+1] = set(flatten([directed[a] for a in steps[i] if a in directed.keys() if a not in visited]))
        visited.update(steps[i])
        steps[i+1] = steps[i+1] - steps[i+1].intersection(visited)
        i+=1
        if i > stop_level:
            print('stop level reached')
            break
    count = (list(steps.keys())[-2])
    
    reachable_nodes = list(set.union(*steps.values()))
    aux = list(steps.values())
    # compute total distance (unitary edges)
    
    distance = 0
    for i in range(len(aux)):
        distance += len(aux[i]) * (i+1)
    
    '''
    first tentative
    distance=0
    for node in tqdm(reachable_nodes):
        for i in range(len(aux)):
            if node in aux[i]:
                distance += i+1
                break
    '''
    
    clos_cent = (len(reachable_nodes)-1) / distance
    
    print(f"Starting from the reference node (user) {source} and following a directional path, it is possible to reach {len(reachable_nodes)} of {len(nodes)} nodes (users).")
    print(f"The farthest reachable node is {count} edges (unitary steps) far from the reference node.")
    print(f"Closeness centrality of the reference node is equal to {round(clos_cent,5)}")
    
    return steps, round(clos_cent, 5), reachable_nodes

def closeness_dijkstra(source, edges, directed, nodes):
    # compute closeness by using function find_paths
    distances, _, _ = find_paths(source, edges, directed, nodes)
    dist = np.array(list(distances.values()))
    dist = dist[dist < INT_MAX]
    clos_centr = len(dist)/dist.sum()
    return round(clos_centr, 5)

def pagerank(source, edges, nodes):
    edges_arr = np.array(list(edges.keys()))
    data = np.ones(len(edges_arr), dtype = 'int8')
    rows = edges_arr[:,0]
    cols = edges_arr[:,1]
    
    #dict to trace back users
    code_nodes = defaultdict()
    codes = list(enumerate(list(nodes)))
    for i in range(len(nodes)):
        code_nodes[codes[i][1]] = i
    
    for j in range(len(edges_arr)):
        edges_arr[j][0] = code_nodes[edges_arr[j][0]]
        edges_arr[j][1] = code_nodes[edges_arr[j][1]]
    
    # define adjiacency matrix
    M = sp.sparse.coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)), dtype='int8')
    
    # out degree using adj matrix
    S = np.array(M.sum(axis=1)).flatten()
    # diagonal matrix (inverse out degree)
    S[S != 0] = 1.0 / S[S != 0]
    
    Q = sp.sparse.spdiags(S.T, 0, *M.shape, format="csr")
    P = Q * M
    
    N = len(nodes)
    x0 = np.repeat(0 / N, N)
    x1 = np.repeat(1.0 / N, N)
    beta = np.repeat(1.0 / N, N)
    alpha = 0.85
    
    for _ in range(100):
        x0 = x1
        x1 = beta + (alpha) * (x1 * P)
        # check convergence, l1 norm
        err = np.absolute(x1 - x0).sum() 
        if err < N * 0.000000000001:
            break
        
    pag_centr = x1[code_nodes[source]]
    print(f'Pagerank centrality of node {source} is equal to {pag_centr}')
    return pag_centr

def find_paths_mid(source, edges, directed, nodes, mid_node):
    # exstension of find_paths function
    n = len(nodes)
    
    # key: node
    distances = defaultdict() # value: shortest distance from source node
    paths = defaultdict() # value: number shortest paths from source node
    visited = defaultdict() # value: boolean
    route = defaultdict() # value: "parent" node closest to the source node
    mid = defaultdict()
    
    for node in nodes:
        distances[node] = INT_MAX 
        paths[node] = 0
        visited[node] = False
    
    distances[source] = 0
    paths[source] = 1
    route[source] = 0
    #mid[source] = 0
 
    q = deque()
    q.append(source)
    visited[source] = True
    
    
    while q:
        current = q[0]
        q.popleft()
        
        # nodes without out-edges
        if current not in directed.keys():
            # add unvisited node to the queue
            if not visited[current]:
                q.append(current)
                visited[current] = True
        
        else:
            # for all neighboring nodes of current one
            for x in directed[current]:
                        
                # add unvisited node to the queue
                if not visited[x]:
                    q.append(x)
                    visited[x] = True
                    route[x] = current
                    
                    if current == mid_node:
                        mid[x] = paths[current]
                    if current in mid.keys():
                        mid[x] = paths[current]

                # check if there is a better path
                if distances[x] > distances[current] + edges[(current, x)]: # +1 unweighted
                    distances[x] = distances[current] + edges[(current, x)] # +1 unweighted
                    paths[x] = paths[current]

                # additional shortest paths found
                elif distances[x] == distances[current] + edges[(current, x)]: # +1 unweightes
                    paths[x] += paths[current]
                    if x in mid.keys():
                        mid[x] += 1 
                            
    return distances, paths, route, mid

def betweeness_centrality(mid_node, edges, directed, nodes): 
    n = len(nodes)
    bet_centr = 0
    costant = (n**2 - 3*n + 2) / 2
    for node in tqdm(list(nodes)):
        if node in directed.keys():
            _, paths, _, mid = find_paths_mid(node, edges, directed, nodes, mid_node)
            for x in mid.keys():
                bet_centr += mid[x]/paths[x]
    return bet_centr/costant

def visualize_graph(source, directed, edges):
    # define subgraph of nodes and its neighbours
    G = nx.DiGraph()
    G.add_node(source)
    for neighbour1 in directed[source]:
        G.add_node(neighbour1)
        G.add_edge(source, neighbour1, weight=edges[(source, neighbour1)])
        if neighbour1 in directed.keys():
            for neighbour2 in directed[neighbour1]:
                if G.has_node(neighbour2):
                    G.add_edge(neighbour1, neighbour2, weight=edges[(neighbour1, neighbour2)])
    
    # plot graph
    val_map = {source: 1.0}
    values = [val_map.get(node, 0.45) for node in G.nodes()]
    edge_labels=dict([((u,v,),d['weight'])
                     for u,v,d in G.edges(data=True)])

    edge_colors = ['black' for edge in G.edges()]

    #edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]

    pos = nx.spring_layout(G, k=0.5, iterations=20)
    plt.figure(3,figsize=(12,12))
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)

    nx.draw(G,pos, node_color = values, node_size=40,edge_color=edge_colors,edge_cmap=plt.cm.Reds)#, node_color = values
    plt.show()

    
# FUNCTIONALITY 4

# BFS but it only tells if a node has been visited
def BFS(s, t, parent):

    visited = []  # all vertices has not been visited

    queue = [s]  # initialize queue with first vertex inside
    visited.append(s)  # set first node as visited

    while queue:
        u = queue.pop(0)  # enqueue u

        if u in g.keys():

            for v_w in g[u]:  # for every neighbor v of u

                v = v_w[0]  # node
                w = v_w[1]

                if v not in visited and w > 0:  # if v has not been visited yet
                    queue.append(v)  # enqueue
                    visited.append(v)  # put it in visited
                    parent[v] = u  # set v as u parent

        elif u not in g.keys() and u not in visited:
            queue.append(u)
            visited.append(u)

    return True if t in visited else False

# DFS but it only finds the path
def DFS(s, g, visited):

    visited.append(s)

    for u in g:  # for every node in the graph
        for v in g[u]:  # for every neighbor of u
            if v[1] > 0 and v[0] not in visited:  # if it has not been visited yet
                DFS(v[0], g, visited)  # run DFS on it


def minCut(s, t, g):

    cutValue = 0
    maxFlow = 0

    parent = {k:-1 for k in g}

    # While there exists a path between the two nodes
    # augment it
    while BFS(s, t, parent):

        # Find the maximum flow through the path found
        # going from the the sink to the source
        pathFlow = float('inf')
        v = t
        while v != s:

            # Find the edge linking the node with its parent
            for u_w in g[parent[v]]:
                u = u_w[0]
                if u == v:

                    # Take the the bottleneck edge
                    pathFlow = min(pathFlow, u_w[1])
                    # And continue with its parent
                    v = parent[v]
                    break

        # Add path flow to overall flow
        maxFlow += pathFlow

        # Update residual capacities of the edges and reverse edges
        # along the path
        v = t
        while v != s:
            u = parent[v]

            # loof for v in u
            for i in range(len(g[u])):
                k = g[u][i][0]
                wk = g[u][i][1]

                if k == v and wk > 0:

                    g[u][i][1] = wk - pathFlow

                    if len(g[k]) > 0:
                        # add flag to see if it find the child
                        found = False

                        # look for u in v
                        for j in range(len(g[k])):
                            p = g[k][j][0]
                            wp = g[k][j][1]

                            if p == u and wp > 0:
                                g[k][j][1] = wp + pathFlow
                                found = True
                                break
                            else:
                                continue

                        if not found:
                            g[k].append([u, pathFlow])

                    else:
                        g[k].append([u, pathFlow])
                        break

            v = parent[v]

    # Use DFS to find the path
    visited = []
    DFS(s, g, visited)

    # Count the number of edges that initially had weights
    # but now do not
    nodes = []
    for u in g:
        for v in g[u]:
            if v[1] == 0 and u in visited:
                print(u, '-', v[0])
                nodes.append(tuple([u,v[0]]))
                cutValue += 1

    return cutValue, nodes

def visualization_4(node_1, node_2, g):
  
    all_nodes = []
    G = nx.DiGraph()
    list_input = [int(node_1), int(node_2)]
    
    for node in list_input:
     # the nodes you want to disconnect
        all_nodes.append(node)
        print(all_nodes)
        
        for child_weight in g[node]:  # all children of the nodes you want to disconnect
            child = child_weight[0]
            all_nodes.append(child)
            G.add_edge(node, child)

    pos = nx.layout.spring_layout(G)

    nx.draw_networkx_nodes(G, nodelist=all_nodes, node_color = '#D90368', pos=pos, node_size=500)
    nx.draw_networkx_nodes(G, nodelist=nodes[0], node_color = 'green', pos=pos, node_size=500)

    plt.tight_layout()
    plt.legend(scatterpoints = 1, markerscale = 0.2)
    color_map = ['#D90368' if node not in nodes[0] else 'green']
    nx.draw(G, with_labels = True, node_color=color_map, alpha = 0.4, pos = pos, arrowsize=15, node_size = 600)
    nx.draw_networkx_edges(G, edgelist=nodes, width=2.0, pos=pos, edge_color='green')