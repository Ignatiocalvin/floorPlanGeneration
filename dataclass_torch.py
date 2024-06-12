import torch
import torch.nn as nn
import torch_geometric as pyg
from torchvision import transforms
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import os
import numpy as np
from utils import load_pickle

transform = transforms.ToTensor()
target_transform = lambda x: torch.from_numpy(np.array(x, dtype=np.int64))

class preprocessDataset(torch.utils.data.Dataset):
    def __init__(self, path, graph_type = 'zoning'):
        self.graph_path = os.path.join(path, 'graph_in' if 'zoning' in graph_type else 'graph_out')
        self.struct_path = os.path.join(path, 'struct_in')
        self.full_path = os.path.join(path, 'full_out')

        # For boundary image to tensors
        self.transform = transform
        # For segmentation mask to tensors
        self.target_transform = target_transform
        
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.graph_path)]

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        graph_nx = load_pickle(os.path.join(self.graph_path, f'{filename}.pickle'))
        struct_in = np.load(os.path.join(self.struct_path, f'{filename}.npy'))
        full_out = np.load(os.path.join(self.full_path, f'{filename}.npy'))
        graph_nx.graph['struct'] = struct_in[np.newaxis, ...]
        graph_nx.graph['full'] = full_out[np.newaxis, ...]

        # Load boundary image
        boundary_image = struct_in.astype(np.uint8)
        boundary_image = self.transform(boundary_image)
        # torch.Size([3, 512, 512])
        
        # Load ground truth image
        gt_image = full_out[..., 0].astype(np.uint8)
        gt_image = self.target_transform(gt_image)
        
        # rooms are divided into 4 types:
        # 'Zone1': ['Bedroom'],
        # 'Zone2': ['Livingroom', 'Kitchen', 'Dining', 'Corridor'],
        # 'Zone3': ['Stairs', 'Storeroom', 'Bathroom'],
        # 'Zone4': ['Balcony'],
        # these are the nodes in the graph

        # 3 connection types:
        # 'door', 'entrance door', 'passage'
        # these are the edges in the graph
        num_room_types = 4  
        num_connection_types = 3 
        connection_dic = {'door': 0, 'entrance': 1, 'passage': 2}
        
        # For node attributes
        # They look like: 
        # NodeDataView({0: {'zoning_type': 1}, 1: {'zoning_type': 2}, 2: {'zoning_type': 1}, 3: {'zoning_type': 1} ...
        # So we make a one hot encoding for each node. 
        # E.g. [0, 1, 0, 0] for first node since it has zoning_type 1 and [0, 0, 1, 0] for second node since it has zoning_type 2
        node_features = []
        for _, node_data in graph_nx.nodes(data=True):
            node_type = node_data['zoning_type']
            node_feature = [0]*num_room_types
            node_feature[node_type] = 1
            node_features.append(node_feature)

        # For edge attributes
        # They look like:
        # EdgeDataView([(0, 1, {'connectivity': 'door'}), (0, 2, {'connectivity': 'door'}), (0, 5, {'connectivity': 'passage'}) ...
        # So we make a one hot encoding for each edge.
        # E.g. [1, 0, 0] for first edge since it has connectivity 'door' and [0, 0, 1] for third edge since it has connectivity 'passage'
        edge_features = []
        for _, _, edge_data in graph_nx.edges(data=True):
            connection_type = connection_dic[edge_data['connectivity']]
            edge_feature = [0]*num_connection_types
            edge_feature[connection_type] = 1
            edge_features.append(edge_feature)

        # Convert to PyG graph
        graph_pyg = pyg.utils.from_networkx(graph_nx)
        graph_pyg.x = torch.tensor(node_features, dtype=torch.float)  # node features
        graph_pyg.edge_attr = torch.tensor(edge_features, dtype=torch.float)  # edge features

        return boundary_image, graph_pyg, gt_image

    def __len__(self):
        return len(self.filenames)
    
class GraphFloorplanUNet(nn.Module):
    def __init__(self, num_node_features, in_channels, out_channels):
        super(GraphFloorplanUNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # Upsampling
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # torch.Size([8, 256, 64, 64])
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # Upsampling
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # Upsampling
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        )
        
        self.floorplan_gnn = FloorPlanGNN(num_node_features)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, graph_data):
        
        # Forward pass through the encoder
        for layer in self.encoder:
            x = layer(x)
        # torch.Size([16, 256, 32, 32])

        # Forward pass through the graph neural network
        graph_features = self.floorplan_gnn(graph_data)
        # Graph Feature Shape: torch.Size([16, 256, 32, 32])
        
        # Concatenate along channel dimension
        x = torch.cat([x, graph_features], 1)  
        # torch.Size([16, 512, 32, 32])

        # Forward pass through the decoder
        for layer in self.decoder:
            x = layer(x)

        # To 11 classes
        return self.final_conv(x) 

# Graph Neural Network
class FloorPlanGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(FloorPlanGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 1024)
        self.fc = nn.Linear(1024, 512*512)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # X is the one hot encoded node features
        # Shape: torch.Size([209, 4])

        # Edge Index Shape: torch.Size([2, 432])
        # Edge Index gives us the node pairs that are connected

        # Batch is the batch index for each node

        x = self.conv1(x, edge_index)
        # Conv1 Result Shape: torch.Size([226, 64])
        x = F.relu(x)
        x = F.dropout(x)
        # Dropout Result Shape: torch.Size([226, 64])
        x = self.conv2(x, edge_index)
        # Conv2 Result Shape: torch.Size([226, 1024])
        x = global_mean_pool(x, batch)
        # Global Mean Pool Result Shape: torch.Size([8, 1024])
        x = self.fc(x).view(-1, 256, 32, 32)
        # FC Result Shape: torch.Size([8, 256, 32, 32])
        return x