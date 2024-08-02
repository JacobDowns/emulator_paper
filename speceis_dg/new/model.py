import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
import torch 

def build_mlp(in_size, hidden_size, out_size, lay_norm=False):

    module = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size)
    )
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module


def copy_geometric_data(graph):

    x_edge = graph.x_edge
    x_node = graph.x_node
    edge_index = graph.edge_index
    
    ret = Data(x_edge = x_edge, x_node = x_node, edge_index=edge_index)
    
    return ret


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=32,
                hidden_size=32):
        super(Encoder, self).__init__()

        #self.eb_encoder = nn.Linear(edge_input_size, hidden_size) 
        #self.nb_encoder = nn.Linear(hidden_size, hidden_size) #build_mlp(hidden_size, hidden_size, hidden_size)

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(hidden_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        x_edge = graph.x
        edges = graph.edge_index
        x_edge_ = self.eb_encoder(x_edge)

        x_node = torch.zeros((graph.num_nodes, x_edge_.shape[1]), device=x_edge.device)
        scatter_add(x_edge_, edges[:,0], dim=0, out=x_node)
        scatter_add(x_edge_, edges[:,1], dim=0, out=x_node)

        x_node_ = self.nb_encoder(x_node)
     
        return Data(x_node = x_node_, x_edge = x_edge_, edge_index=edges)


class EdgeBlock(nn.Module):

    def __init__(self, custom_func=None):
        
        super(EdgeBlock, self).__init__()
        self.net = custom_func


    def forward(self, graph):

        x_edge = graph.x_edge
        x_node = graph.x_node
        edges = graph.edge_index

        x_edge_ = torch.cat([
            x_node[edges[:,0]],
            x_node[edges[:,1]],
            x_edge
        ], dim=1)

        x_edge_ = self.net(x_edge_)  

        return Data(x_edge=x_edge_, x_node=x_node, edge_index=edges)


class NodeBlock(nn.Module):

    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, graph):

        x_node = graph.x_node
        x_edge = graph.x_edge
        edges = graph.edge_index

        # Aggregate edge features
        x_node1 = torch.zeros((x_node.shape[0], x_edge.shape[1]), device=x_node.device)
        scatter_add(x_edge, edges[:,0], dim=0, out=x_node1)
        scatter_add(x_edge, edges[:,1], dim=0, out=x_node1)

        x_node_ = torch.cat([
            x_node,
            x_node1
        ], dim=1)

        x_node_ = self.net(x_node_)
        return Data(x_node=x_node_, x_edge=x_edge, edge_index=edges)
       

class GnBlock(nn.Module):

    def __init__(self, hidden_size=32):

        super(GnBlock, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        x_edge = graph_last.x_edge + graph.x_edge
        x_node = graph_last.x_node + graph.x_node
        return Data(x_node=x_node, x_edge=x_edge, edge_index=graph.edge_index)


class Decoder(nn.Module):

    def __init__(self, hidden_size=32, output_size=1):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)
        #self.decode_module = nn.Linear(hidden_size, output_size)

    def forward(self, graph):
        return self.decode_module(graph.x_edge)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, edge_input_size=9, hidden_size=64):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, hidden_size=hidden_size)
        
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=3)

    def forward(self, graph):

        graph = self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded







