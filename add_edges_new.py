import pickle
import dgl
import torch

infile = open('texts', 'rb')
texts = pickle.load(infile)
infile.close()

g = dgl.DGLGraph()
g.add_nodes(len(texts))

current_nodes = []
pretext = ''
for i in range(len(texts)):
    text = texts[i]
    if text != pretext:
        # add edge
        for j in current_nodes:
            g.add_edges(j, current_nodes)
        current_nodes = []
        pretext = text
    current_nodes.append(i)
    pretext = text

for j in current_nodes:
    g.add_edges(j, current_nodes)

file_name = 'graph_add_edge'
outfile = open(file_name, 'wb')
pickle.dump(g,outfile)
outfile.close()

print(g.number_of_nodes())
print(g.number_of_edges())
