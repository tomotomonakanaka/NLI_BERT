import pickle
import dgl
import torch

infile = open('graph','rb')
g = pickle.load(infile)
infile.close()

infile = open('texts', 'rb')
texts = pickle.load(infile)
infile.close()

edge_features = [1] * g.number_of_edges()

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

edge_features += [2] * (g.number_of_edges()-len(edge_features))
edge_features = torch.Tensor(edge_features)
print(g.number_of_edges())
print(edge_features.shape)

file_name = 'edge_features'
outfile = open(file_name, 'wb')
pickle.dump(edge_features,outfile)
outfile.close()
