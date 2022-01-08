import enum
import torch
from torch import nn
import numpy as np
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision import transforms, datasets

# Decide whether or not to use GPU based on CUDA availability
device = torch.device(type='cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Constants
CHANNELS = 3
EPOCHS = 100

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor()
])

# Construct a DataLoader out of the root image directory
full_loader = DataLoader(datasets.ImageFolder('images', transform=transform), batch_size=1, shuffle=True)

# Split the DataLoader's Dataset into training and validation DataLoaders
train_len = int(0.75*len(full_loader))
test_len = len(full_loader) - train_len

train_data, test_data = random_split(full_loader.dataset, [train_len, test_len])
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# Define the classifier network
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=CHANNELS,
                out_channels=64,
                kernel_size=3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )

        self.flatten = nn.Flatten()

        self.dense_stack = nn.Sequential(
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
            nn.LazyLinear(out_features=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        return self.dense_stack(x)

# Create the model and assign it to the current device
model = Classifier().to(device)

# Define the loss function and the optimizer (no logits for BCE because we already applied sigmoid)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train():
    for epoch in range(EPOCHS):
        # Metrics
        cumulative_loss = 0
        train_acc = 0
        test_acc = 0
        for data in train_loader:

            # Unpack data into inputs and labels
            images, labels = data[0].to(device), data[1].to(device)

            # Initialize the gradients
            optimizer.zero_grad()

            # Get the output from the model
            pred_labels = model(images)
            
            # Calculate the loss; first arg is ŷ and second is y
            labels = labels.unsqueeze(1).float()

            # Calculate the accuracy during training
            for i, v in enumerate(pred_labels):
                if torch.round(pred_labels[0][i]) == labels[0][i]:
                    train_acc += 1

            # Calculate the loss and add it to the running loss
            loss = loss_function(pred_labels, labels)
            cumulative_loss += loss.item()

            # Do backprop to get the gradient
            loss.backward()

            # Iterate through the parameters and update their weights
            optimizer.step()

        # Validation loop
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)

            pred_labels = model(images)
            labels = labels.unsqueeze(1).float()

            for i, v in enumerate(pred_labels):
                if torch.round(pred_labels[0][i]) == labels[0][i]:
                    test_acc += 1
        
        # Print the metrics
        print(f'EPOCH {epoch+1} || Loss: {cumulative_loss/len(train_loader)} || Train acc: {train_acc/len(train_loader)} || Test acc: {test_acc/len(test_loader)}')

train()