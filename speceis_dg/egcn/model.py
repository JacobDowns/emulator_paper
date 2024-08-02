import torch.nn as nn
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.data import Data
import numpy as np


        
def build_mlp(in_size, hidden_size, out_size, lay_norm=False):

    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module


def create_mlp(input_dim, output_dim, hidden_dims, activation_fn=nn.ReLU, output_activation=None):
    """
    Creates an MLP with the given dimensions.

    Parameters:
    - input_dim (int): Dimension of the input layer.
    - output_dim (int): Dimension of the output layer.
    - hidden_dims (list of int): List of dimensions of the hidden layers.
    - activation_fn (callable): Activation function to use for hidden layers. Default is nn.ReLU.
    - output_activation (callable or None): Activation function to use for the output layer. Default is None.
    
    Returns:
    - nn.Sequential: The constructed MLP.
    """
    layers = []
    in_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(activation_fn())
        in_dim = h_dim

    layers.append(nn.Linear(in_dim, output_dim))

    if output_activation is not None:
        layers.append(output_activation())

    return nn.Sequential(*layers)


def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr = graph.node_attr 
    edge_index = graph.edge_index 
    edge_attr = graph.edge_attr 

    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    
    return ret


class EdgeBlock(nn.Module):

    def __init__(self, custom_func=None):
        
        super(EdgeBlock, self).__init__()
        self.net = custom_func


    def forward(self, graph):

        node_attr, edge_index, edge_attr, _ = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)
        
        edge_attr_ = self.net(collected_edges)   # Update

        return Data(coords = graph.coords, x=node_attr, edge_attr=edge_attr_, edge_index=edge_index)


class EGCN(nn.Module):

    def __init__(
            self,
            h_size,
            a_size,
            h_star_size,
            a_star_size,
            hidden_size
    ):

        super(EGCN, self).__init__()

        self.h_size = h_size 
        self.a_size = a_size 
        self.hidden_size = hidden_size
        
        self.phi_e = create_mlp(2*h_size + a_size + 1, hidden_size, [hidden_size])
        self.phi_h = create_mlp(2*h_size + a_size + 1, hidden_size, [hidden_size])
        self.mlp1 = create_mlp(hidden_size, hidden_size, [hidden_size])
        self.mlp2 = create_mlp(2*hidden_size, hidden_size [hidden_size])

    def forward(self, graph):

        # Vertex coordinates
        x = graph.x 
        # Vertex features 
        h = graph.h 
        # Edges 
        edges = graph.edges 
        # Node to node vector
        dx = (x[edges[:,1]] - x[edges[:,0]])
      
        # Dual mesh vertex coordinates
        x_star = graph.x_star
        # Dual mesh vertex features
        h_star = graph.h_star 
        # Dual mesh edges
        edges_star = graph.edges_star 
        # Node to node vector
        dx_star = (x_star[edges_star[:,1]] - x_star[edges_star[:,0]])

        cell_to_vertex_map = graph.cell_to_vertex_map 
        vertex_to_cell_map = graph.vertex_to_cell_map 

        ### Standard mesh messages
        #######################################################

        # Compute messages
        m_features =  []
      
        m_features = [
            h[edges[:,0]],
            h[edges[:,1]],
            np.linalg.norm(dx, dim=1)
        ]

        if 'a' in graph:
            m_features.append(graph.a) 
        
        m_edge = self.phi_e(m_features)

        # Coordinate update 
        x1 = x + scatter_mean(dx, edges[:,1], dim=0) 
        # Aggregate messages 
        m_node = scatter_add(m_edge, edges[:,1], dim=0)

        
        ### Dual mesh messages
        #######################################################

        m_star_features =  [
            h_star[edges_star[:,0]],
            h_star[edges_star[:,1]],
            np.linalg.norm(dx_star, dim=1)
        ]

        if 'a_star' in graph:
            m_star_features.append(graph.a_star) 

        m_star_edge = self.phi_star_e(m_star_features)

        # Coordinate update 
        x_star_1 = x_star + scatter_mean(dx_star, edges_star[:,1], dim=0) 
        # Aggregate messages 
        m_star_node = scatter_add(m_star_edge, edges_star[:,1], dim=0)


        ### Messages from cells to vertices 
        ############################################

        dx0 = x[vertex_to_cell_map[:,0]] - x_star[vertex_to_cell_map[:,1]]

        m_v_c_edge = self.phi_v_n([
            h_star[vertex_to_cell_map[:,1]],
            torch.linalg.norm(dx0, dim=1)
        ])

        # Aggregate the messages
        m_v_c = scatter_add(m_v_c_edge, vertex_to_cell_map[:,0], dim=0)


        ### Messages from vertices to cells
        ############################################

        dx1 = x_star[cell_to_vertex_map[:,0]] - x[cell_to_vertex_map[:,1]]

        m_c_v_edge = self.phi_n_v([
            h[cell_to_vertex_map[:,1]],
            torch.linalg.norm(dx1, dim=1)
        ])

        # Aggregate messages 
        m_c_v = scatter_add(m_c_v_edge, cell_to_vertex_map[:,0], dim=0)   

        ### New node features on mesh and dual mesh
        ############################################

        h1 = self.phi_h(m_node, m_v_c)
        h_star1 = self.phi_h_star(m_star_node, m_c_v)



        return Data(pos=x1, node_attr=h1, edge_attr=a1, edge_index=edge_index)
    

class MultiArgMLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):

        super(NodeBlock, self).__init__()
        self.net = create_mlp(in_size, out_size, [hidden_size])

    def forward(self, *args):
        features = []
        for arg in args:
            features.append(arg)

        features = torch.cat(features, dim=1)
        return self.net(features)

    
class NodeBlock(nn.Module):

    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph.edge_attr
        nodes_to_collect = []
        
        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        x = self.net(collected_nodes)
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)
       
            
            


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
        
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)



class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

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
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)



class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size)
        
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):

        graph= self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded