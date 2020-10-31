#assign shorter alias to networkx so I can type a short word when
#calling one of its methods
#eg nx.dfs_edges
import networkx as nx

#shorter alias for numpy
import numpy as np

import math

# seed random number generator
#np.random.seed(1)

#the graph
global G

#create a directed graph object
G = nx.DiGraph()

output_verbose = True
output_interactive = False

print("Press 1 for random edges generation or 2 to input nodes/ edges from file")
mode_input = int(input())

if mode_input==1:
    #read number of nodes from user
    print("How many nodes does the graph have?")
    num_nodes = int(input())    
    #add nodes to graph according to num nodes
    gnodes = []
    for node in range(num_nodes):
        gnodes.append(node)
    
    print("gnodes=", gnodes)    
    G.add_nodes_from(gnodes)

    #gedges = array of graph edges (will be added one by one with a loop)
    #gedges =[(0, 1), (0, 3), (1, 2), (0, 2), (3, 2), (2, 4), (1, 4), (3, 8), (5, 6), (6, 7), (5, 7), (7, 8), (4, 5), (5,4)]

    #number of nodes in graph
    n = G.number_of_nodes()

    #average node degree 2m/n
    #2m edges
    #generate edges by random draw
    gedges = []
    
    m = n*math.log(n) - 1
    for i in range(int(m)):    
        edge_nodes = []
        edge_nodes = np.random.choice(list(gnodes), 2, replace=False)
        found = False
        #check if there is already an edge between the two nodes
        for i in range(len(gedges)):
            if gedges[i][0]==edge_nodes[0] and gedges[i][1]==edge_nodes[1]:
                found = True
                break
        while found == True:
            edge_nodes = []
            edge_nodes = np.random.choice(list(gnodes), 2, replace=False)
            found = False
            for i in range(len(gedges)):
                if gedges[i][0] == edge_nodes[0] and gedges[i][1] == edge_nodes[1]:
                    found = True

        gedges.append(edge_nodes)

    print("gedges = ", gedges)

if mode_input==2:
    print("reading from file")
    gnodes = []
 
    #read edges from file
    gedges = []

    lineCounter = 0
    #inFile = open("graph2.txt", "r")
    inFile = open("graph.txt", "r")
    for fRow in inFile:
        lineCounter = lineCounter + 1
        if lineCounter == 1:
            n = int(fRow.split()[0])
            m = int(fRow.split()[1])
            for node in range(n):
                gnodes.append(node)
                G.add_nodes_from(gnodes)
            #gedges = []
            print(n,m)
        else:
            gedges.append([int(fRow.split()[0]), int(fRow.split()[1])])
    inFile.close()

    print("gedges=", gedges)    


#initialize empty dictionaries for ancestors and descendants
#the dictionary keys will be the selected/ randomnly generted nodes
#and the values will be the corresponding lists
ancestors = dict()
#As = ancestors intersect S for all nodes in S
As = dict()
#set Ai is a subset of ancestors based on S equivalency
Ai = dict()
descendants = dict()
#Ds = descendants intersect S for all nodes in S
Ds = dict()
parents = dict()
#parentsA = dict()

#generate random node ids
#empty set
current_nodes = set()
current_nodes = G.nodes
num_current_nodes = n

#generate S* set of nodes
global S
S = set()
Slen = round(math.log(n)*math.sqrt(n))

S = np.random.choice(list(current_nodes), Slen, replace=False)
#S = {0,1,2}
#Slen = 3
#print("S = ", S)
#S = G.nodes
#Slen = n
#S =  [0, 2, 6, 5, 1]
print("S = ", S)


#add node keys to results sets, initially with empty lists
#only parents initialized    
for node in current_nodes:
    ancestors.setdefault(node, [])
    As.setdefault(node, [])
    Ai.setdefault(node, [])
    descendants.setdefault(node, [])
    Ds.setdefault(node, [])
    parents.setdefault(node, [])
    
    if node in S:
        #add nodes in S to all their sets
        Ds[node].append(node)
        As[node].append(node)
           
        descendants[node].append(node)
        ancestors[node].append(node)
        
    #all parents initialized to -1
    for node2 in range(n):
        parents[node].append(-1)
        

################functions for node equivalency#########################################
#returns 1 if nodes are equivalent
def nodes_equivalent(node1, node2):
    if len(ancestors[node1]) == len(ancestors[node2]) and len(descendants[node1])==len(descendants([node2])):
        return 1
    else:
        return 0

#returns 1 if nodes are S equivalent
def nodes_S_equivalent(node1, node2):
    #if (node1 not in Ds[node2]) and (node1 not in As[node2]):
     #   return 0
    if len(As[node1]) == len(As[node2]) and len(Ds[node1])==len(Ds[node2]):
        return 1
    else:
        return 0
  
#returns 1 if cycle detected when inserting edge u, v
def explore(u, v):
    if output_verbose:
        print("calling explore for edges", u, v)
    to_explore = set()
    to_explore.add(v)
    while len(to_explore)>0:
        
        w = to_explore.pop()
        print("currently reached ", w)
        if w==u:
            print("reached the start --> cycle detected")
            return 1
        if w in Ai[u]:
            print(w, "in Ai[",u,"]", Ai[u])
            return 1
        else:
            if u in Ai[w]:
                print(u, "in Ai[",w,"]", Ai[w])
            if nodes_S_equivalent(u, w)==0:
                if output_verbose:
                    print(u,w,"not s equivalent")
            else:
                if output_verbose:
                    print(u,w,"ARE s equivalent")
                Ai[w].append(u)
                    
                for e in gedges:
                    if e[0]==w:
                        to_explore.add(e[1])
                                    
    return 0
#########################################################
#section for topologicalOrdering
#global topologicalOrdering
#topologicalOrdering = []
global L
L = []
buckets = dict()
placeholders = dict()
#up and down for each bucket --> dictionaries with two keys
up = dict()
down = dict()
#up and down for all buckets --> lists
#up_all = []/
#down_all = []

#dictionary with two keys: number of ancestors intersect S, number of descendants intersect S
#Slen = sqrt(n)*log(n) -> calculated before
#buckets_values = tuples of keys i, j
#i>=0
#j<=12 log(n)*n
#j_limit = 12*Slen + 1

i_limit = Slen + 1
j_limit = Slen + 1

#keys for dictionaries (tuples)
dicts_keys = []
for i in range(i_limit):
    for j in range(j_limit):
        dicts_keys.append((i,j))
#if output_verbose:
    #print(dicts_keys)
#the tuple is the key    
for tupleT in dicts_keys:
    buckets.setdefault(tupleT, [])
    placeholders.setdefault(tupleT, -1)
    up.setdefault(tupleT, [])
    down.setdefault(tupleT, [])


#placeholder elements in list L
placeInList = 0
#for every bucket
for i in range(i_limit):
    for j in range(j_limit): 
        placeholders[(i,j)] = [-1 - placeInList]
        L.append(placeholders[(i,j)][0])
        placeInList = placeInList + 1


#print("placeholders=", placeholders)
#print("L=",L)

#the set of nodes currently added to buckets
#will be used when adding edges
#all nodes added initially
#nodes_in_buckets = set()
nodes_added = set()

#Insert-Before(X; Y ): given a pointer to X, insert
#element Y immediately before element X.
def listInsertBefore(X, Y, L):
    i = L.index(X)
    L.insert(i, Y)
    #insert method inserts BEFORE given position
    return 0

def listInsert(X, Y, L):
    i = L.index(X)
    L.insert(i+1, Y)
#list.insert(i, x)
#Insert an item at a given position. The first argument is the index of the
#element before which to insert, so a.insert(0, x) inserts at the front of
#the list, and a.insert(len(a), x) is equivalent to a.append(x).
    return 0

def delete(X, L):
    L.remove(X);
    #if I need the element returned L.pop(L.index(X))
    return 0

def order(X, Y, L):
    #returns true if X before Y, false otherwise
    return (L.index(X)<L.index(Y))

#returns i,j indices for bucket(node)
def getBucket(node):
    acounter = len(As[node])
    dcounter = len(Ds[node])

    i = acounter
    j = Slen - dcounter
    return i,j


#sort elements of arr based on their order in Lst
# using quicksort

#first partition array to be sorted based on a pivot element
def partition(arr, l, h, Lst): 
    i = ( l - 1 ) 
    x = arr[h] 
  
    for j in range(l , h): 
        if   Lst.index(arr[j]) <= Lst.index(x): 
          #comparison based on positions in Lst
            # increment index of smaller element 
            i = i+1
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[h] = arr[h],arr[i+1] 
    return (i+1) 
  
# Function to do Quick sort without recursion
# arr[] --> Array to be sorted, 
# l  --> Starting index, 
# h  --> Ending index
# Lst --> list containing array elements in desired sort order
def quickSort(arr, Lst): 

    l = 0
    h = len(arr) - 1
    # Create an auxiliary stack 
    size = h - l + 1
    stack = [0] * (size) 
  
    # initialize top of stack 
    top = -1
  
    # push initial values of l and h to stack 
    top = top + 1
    stack[top] = l 
    top = top + 1
    stack[top] = h 
  
    # Keep popping from stack while is not empty 
    while top >= 0: 
  
        # Pop h and l 
        h = stack[top] 
        top = top - 1
        l = stack[top] 
        top = top - 1
  
        # Set pivot element at its correct position in 
        # sorted array 
        p = partition( arr, l, h, Lst ) 
  
        # If there are elements on left side of pivot, 
        # then push left side to stack 
        if p-1 > l: 
            top = top + 1
            stack[top] = l 
            top = top + 1
            stack[top] = p - 1
  
        # If there are elements on right side of pivot, 
        # then push right side to stack 
        if p+1 < h: 
            top = top + 1
            stack[top] = p + 1
            top = top + 1
            stack[top] = h 
  

#insert all graph nodes in list after the 0, 12Slen bucket placelolder
pl = placeholders[(0, Slen)][0]
for node in gnodes:
    listInsert(pl, node, L)

#print("L=",L)
  
#end section for topologicalOrdering
#nodes_added = set()
global cycleDetected
cycleDetected = False

for e in gedges:
    start = e[0]
    end = e[1]
    print(" ")
    print("adding edge ", start, end, "*************************")
    
    #empty the set of affected nodes and add start and end to set of nodes whose buckets I will (re)compute   
    nodes_added.add(start)
    nodes_added.add(end)
    
    if cycleDetected==True:
        print("i am breaking because of cycle")
        break
    
    cycleDetected = False
    G.add_edge(start, end)
    
    #update relevant sets for all tracked nodes
    for current_node in current_nodes:
        
        #special case of tree root
        if start == current_node:
            parents[current_node][start] = start

        #case 1 in e-mail (start is not accessible from the root) --> do nothing
        if parents[current_node][start] == -1:
            continue

        #case 2 in e-mail
        #case 2a in e-mail (end was already in descendants tree) --> do nothing
        if parents[current_node][end] != -1:
            continue

        #case 2b in e-mail (end needs to be added to descendants tree and its parent needs to be updated)
        parents[current_node][end] = start
        if current_node in S:
            descendants[current_node].append(end)
        
        if end in S:
            Ds[current_node].append(end)

        if end in S:
            ancestors[end].append(current_node)

        if current_node in S:
            As[end].append(current_node)

        if nodes_S_equivalent(current_node, end):
            Ai[end].append(current_node)

        #dfs preorder from end to find additional nodes accessible from end
        #print("DFS preorder from ", end)
        dfs_reachable = list(nx.dfs_preorder_nodes(G,end))
        
        #append all nodes reachable from the new edge end to descendants of tree root (0)
        #print(dfs_reachable)
        for node in dfs_reachable:
            #print("DFS node examined: ", node)

                
            #I must append only those that were not already in the tree
            if parents[current_node][node] == -1:
                parents[current_node][node] = end

                if current_node in S:
                    descendants[current_node].append(node)
                
                if node in S:
                    Ds[current_node].append(node)

                if node in S:        
                    ancestors[node].append(current_node)
                    
                if current_node in S:
                    As[node].append(current_node)
                    
               
    if explore(e[0],e[1])==1:
            print("while adding edge (", start, ",", end, ") I explored (",e[0],",",e[1],")","cycle detected")
            cycleDetected = True
            
            if output_verbose:
                print("Ancestors=",ancestors)
                print("Descendants=",descendants)
                print("As=",As)
                print("Ds=",Ds)
                print(" ")
            break
    else:
            print("while adding edge (", start, ",", end, ") I explored (",e[0],",",e[1],")","cycle not detected")


    if output_verbose:
        print("Ancestors=",ancestors)
        print("Descendants=",descendants)
        print("As=",As)
        print("Ds=",Ds)
        print("Ai=",Ai)
        print(" ")
       
    #top ordering    
    #add nodes to buckets only if cycle not detected
    if cycleDetected == False:
        #empty buckets/
        
        buckets_old = buckets

        #empty dictionaries            
        buckets = dict()
        up = dict()
        down = dict()

        for nodeT in dicts_keys:
            buckets.setdefault(nodeT, [])
            up.setdefault(nodeT, [])
            down.setdefault(nodeT, [])
        
        #which bucket is a node currently in?
        
        for node1 in nodes_added:
            #initial values for all nodes
            old_bucket_i = 0
            old_bucket_j = Slen
            for i in range(i_limit):
                for j in range(j_limit):
                    b = buckets_old[(i,j)]
                    for c in range(len(b)):
                        if b[c] == node1:
                            old_bucket_i = i
                            old_bucket_j = j
                            break

            new_bucket_i, new_bucket_j = getBucket(node1)
            if new_bucket_i > old_bucket_i:
                #up_all.add(node1)
                if up[(new_bucket_i, new_bucket_j)].count(node1)==0:
                    up[(new_bucket_i, new_bucket_j)].append(node1)
            if new_bucket_i < old_bucket_i:
                #down_all.add(node1)
                if down[(new_bucket_i, new_bucket_j)].count(node1)==0:
                    down[(new_bucket_i, new_bucket_j)].append(node1)
            if new_bucket_i == old_bucket_i and new_bucket_j > old_bucket_j:
                #up_all.add(node1)
                if up[(new_bucket_i, new_bucket_j)].count(node1)==0:
                    up[(new_bucket_i, new_bucket_j)].append(node1)
            if new_bucket_i == old_bucket_i and new_bucket_j < old_bucket_j:
                #down_all.append(node1)
                if down[(new_bucket_i, new_bucket_j)].count(node1)==0:
                    down[(new_bucket_i, new_bucket_j)].append(node1) 

            for bucket1 in dicts_keys:
                if buckets[bucket1].count(node1)>0:
                    #node found in bucket and must be removed
                    buckets[bucket1].remove(node1)
                    break #at most each node can be in 1 bucket so I do not need to keep looking

            buckets[(new_bucket_i, new_bucket_j)].append(node1)

            #For each non-empty set UPi,j
            #Sort elements in UPi,j in increasing order according to L.
            for i in range(i_limit):
                for j in range(j_limit):
                    if len(up[(i,j)])>0:
                        #sort set according to order in list if more than one elements
                        if len(up[(i,j)])>1:
                            quickSort(up[(i,j)], L)    
                        startI = len(up[(i,j)]) - 1
                        while startI>=0:
                            el = up[(i,j)][startI]
                            delete(el, L)
                            pl = placeholders[(i, j)][0]
                            listInsert(pl, el, L)
                            startI = startI - 1
            #step 3            
            #For each non-empty set DOWNi,j
            #Sort elements in DOWNi,j in increasing order according to L.
            for i in range(i_limit):
                for j in range(j_limit):
                    if len(down[(i,j)])>0:
                        #sort set according to order in list if more than 1 elements
                        if len(down[(i,j)])>1:
                            quickSort(down[(i,j)], L)    
                        startI = 0
                        while startI<len(down[(i,j)]):
                            el = down[(i,j)][startI]
                            delete(el, L)
                            
                            if j<=j_limit:
                                next_bucket_j = j + 1
                                next_bucket_i = i
                            else:
                                next_bucket_i = i + 1
                                next_bucket_j = 0
                            
                            pl = placeholders[(next_bucket_i, next_bucket_j)][0]
                            listInsertBefore(pl, el, L)
                            startI = startI + 1

            #start of step 4                
  
            new_bucket_i_start, new_bucket_j_start = getBucket(start)
            new_bucket_i_end, new_bucket_j_end = getBucket(end)
            
            #start and end in the same bucket

            if new_bucket_i_start==new_bucket_i_end and new_bucket_j_start==new_bucket_j_end:
                to_explore_ts = set()
                to_explore_ts.add(end)
                to_change = []
                while len(to_explore_ts)>0:
                    w = to_explore_ts.pop()
                    if order(start, w, L) == False:
                        #if in different buckets
                        new_bucket_i_w, new_bucket_j_w = getBucket(w)
                        
                        if new_bucket_i_start == new_bucket_i_w and new_bucket_j_start == new_bucket_j_w:
                            to_change.append(w)
                            #4d for every edge (w, z)
                            #add z to explore
                            #check edges already added
                            for ge in gedges:
                                if ge[0] == start and ge[1] == end:
                                    break
                                else:
                                    if ge[0] == w:
                                        to_explore_ts.add(ge[1])
                    #4d
                    if len(to_change)>1:
                        quickSort(to_change, L) 
                    #4e For each w in to_change in decreasing order insert w right after u in L
                    startI = len(to_change) - 1
                    while startI>=0:
                        el = to_change[startI]
                        if (start!=el):
                            delete(el, L)
                            listInsert(start, el, L)
                        startI = startI - 1
        

    if output_verbose:
        print("buckets old = ")
        for i in range(i_limit):
            for j in range(j_limit):
                if len(buckets_old[(i,j)])>0:
                            print("[",i,",",j,"]=",buckets_old[i,j])
        
        print("buckets = ")
        for i in range(i_limit):
            for j in range(j_limit):
                if len(buckets[(i,j)])>0:
                            print("[",i,",",j,"]=",buckets[i,j])
        print("up=")
        for i in range(i_limit):
            for j in range(j_limit):
                        if len(up[(i,j)])>0:
                            print("[",i,",",j,"]=",up[i,j])
        
        print("down=")
        for i in range(i_limit):
            for j in range(j_limit):
                if len(down[(i,j)])>0:
                            print("[",i,",",j,"]=",down[i,j])
    print("Topological order = ")
    for tl in range(len(L)):
        if L[tl]>=0 and (L[tl] in nodes_added):
            print(L[tl])    
    print("finished adding edge ", start, end, "*************************")
    if output_interactive:
        choice = input()
    print(" ")

if cycleDetected == False:
    print("Topological order final = ")
    tl = 0
    while tl<len(L):
        if L[tl]>=0:
            print(L[tl])
        tl = tl + 1
