import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class encoder_decoder(nn.Module): # encoders: sigmoid_output = False, decoder: sigmoid_output = True
    def __init__(self, input_dim, hidden_dim, output_dim, sigmoid_output = False, batch_norm=True):
        super(encoder_decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norm = batch_norm

        input_dim = input_dim # Para que tome las entradas de la primera capa hidden correctamente
        for h in hidden_dim: # hidden dim es una lista con el número de neuronas de cada capa oculta
            self.layers.append(nn.Linear(input_dim, h))
            if self.batch_norm:
                self.layers.append(nn.LayerNorm(h))
            self.layers.append(nn.ReLU())
            input_dim = h # las salidas de esta capa serán las entradas de la siguiente
            
        self.layers.append(nn.Linear(input_dim, output_dim)) # última capa
        if sigmoid_output:
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class gnn(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim_encoder, hidden_dim_decoder, hidden_dim_messagepassing, num_iters=8, batch_norm=True):
        super().__init__(aggr='add')  #usamos suma para agregación de mensajes
        self.num_iters = num_iters # iteraciones del message passing

        self.node_encoder = encoder_decoder(node_in_dim, hidden_dim_encoder, hidden_dim_encoder[-1], batch_norm=batch_norm)
        self.edge_encoder = encoder_decoder(edge_in_dim, hidden_dim_decoder, hidden_dim_encoder[-1], batch_norm=batch_norm) 

        self.edge_messagepassing = encoder_decoder(2*hidden_dim_encoder[-1], hidden_dim_messagepassing, hidden_dim_messagepassing[-1], batch_norm=batch_norm) # el input es el concat de edge acutual + edge0
        self.node_messagepassing = encoder_decoder(3*hidden_dim_encoder[-1], hidden_dim_messagepassing, hidden_dim_messagepassing[-1], batch_norm=batch_norm) # el input es el concat de node actual + node0 + aggregation

        self.decoder = encoder_decoder(3*hidden_dim_messagepassing[-1], hidden_dim_decoder, 1, sigmoid_output=False, batch_norm=batch_norm) # el input es la concatenación de los dos 128 vectores que salen de la parte de edge y de nodes

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h_node = self.node_encoder(x) # x son los nodos, h_node es el embedding de los nodos
        h_edge = self.edge_encoder(edge_attr)

        h_node_0, h_edge_0 = h_node, h_edge # guardamos los embeddings iniciales para la iteración de message passing

        for n in range(self.num_iters): #message passing
            h_edge = self.edge_messagepassing(torch.cat([h_edge, h_edge_0], dim=-1)) # concat h_edge y h_edge_0 que es el inicial, dim=-1 genera un input de tamñao 256 = 2 * 128
            h_node = self.propagate(edge_index, x=h_node, x0=h_node_0, edge_attr=h_edge, size=(x.size(0), x.size(0)))

        node_in_idx, node_out_idx = edge_index # Necesitamos los indices para que al concatenación del input en decoder tengan misma dimension
        return self.decoder(torch.cat([h_node[node_in_idx], h_node[node_out_idx], h_edge], dim=-1))

    #creamos funciones propias para calcular el mensaje y actualizar las respresentaciones
    def message(self, x_j, edge_attr):
        return x_j+ edge_attr
    
    def update(self, aggr_out, x, x0):
        return self.node_messagepassing(torch.cat([x, x0, aggr_out], dim=-1)) # aggr_out es la agregación de los mensajes de los vecinos, x es el embedding del nodo y x0 es el inicial

