import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import os
from datetime import datetime
import pickle
from dataclass_torch import GraphFloorplanUNet
from dataclass_torch import preprocessDataset

# Specify the device to use
device = "cuda:0"

def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    current_time = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join('results', current_time)
    os.makedirs(run_dir)
    
    loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        for images, graph_data, gt_images in train_loader:
            images = images.to(device)
            graph_data = graph_data.to(device)
            gt_images = gt_images.to(device)

            # Zero out the gradients of the parameters before starting to do backpropragation 
            # because PyTorch accumulates the gradients on subsequent backward passes. otherwise it sums the gradients on each backpropagation
            optimizer.zero_grad()

            # Forward pass means computing the predicted outputs by passing the inputs to the model
            outputs = model(images, graph_data)

            # CrossEntropyLoss, which is commonly used for multi-class classification problems. 
            criterion = torch.nn.CrossEntropyLoss()
            
            # Adjust class values
            gt_images[gt_images == 13] = 10
            loss = criterion(outputs, gt_images)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_loader)
        loss_list.append(epoch_loss)

        print(f'Epoch {epoch}, Average Loss: {epoch_loss}')
        
        # Validation
        model.eval()
        val_loss = 0

        # used when you are evaluating a model (i.e., during validation or testing), because you don't need to update 
        # the model's parameters in these phases, so there's no need to compute gradients.
        with torch.no_grad():
            for images, graph_data, gt_images in val_loader:
                images = images.to(device)
                graph_data = graph_data.to(device)
                gt_images = gt_images.to(device)
                
                # Forward pass
                outputs = model(images, graph_data)
                
                # Adjust class values
                gt_images[gt_images == 13] = 10
                loss = criterion(outputs, gt_images)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch}, Average Validation Loss: {val_loss}')
        
        # Save model every 5 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(run_dir, f'model_checkpoint_epoch_{epoch}.pt')
            torch.save(model.state_dict(), save_path)

    with open('loss_list', 'wb') as f:
            pickle.dump(loss_list, f)

num_node_features = 4

"""
'Bedroom': 0,
'Livingroom': 1,
'Kitchen': 2,
'Dining': 3, 
'Corridor': 4,
'Stairs': 5,
'Storeroom': 6,
'Bathroom': 7,
'Balcony': 8,
'Structure': 9,
'Background': 13 -> 10
"""

input_nc = 3
output_nc = 11

# Create the model
model = GraphFloorplanUNet(num_node_features, input_nc, output_nc).to(device)

# Choose a criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# DataLoader
# return boundary_image, graph_pyg, gt_image
train_dataset = preprocessDataset('./dataset/train')
val_dataset = preprocessDataset('./dataset/val')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Train the model
train_model(model, train_loader, optimizer, criterion, epochs=300)
