import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision import models, transforms

class MyData(torch.utils.data.Dataset):
    def __init__(self, data, label, transform):
        self.data = torch.tensor(data).unsqueeze(-1).repeat(1, 1, 1, 3).numpy()
        print(self.data.shape)
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]

    def __len__(self):
        return len(self.label)


    

class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # Assuming input images are 40 x 168


        # self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3, padding=1)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 20 x 84


        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16 , kernel_size=3, padding=1)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 10 x 42


        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 5x 21

        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.relu4 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 2 x 10 

        # self.flattened_size = 32 * 2 * 10

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # for x in self.model.parameters():
        #     x.requires_grad_(False)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 37)
        self.model.fc.requires_grad_(True)
        # self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        # x = x.reshape(x.shape[0], 1, x.shape[1], -1)
        # x = self.pool1(self.relu1(self.conv1(x)))
        # x = self.pool2(self.relu2(self.conv2(x)))
        # x = self.pool3(self.relu3(self.conv3(x)))
        # x = self.pool4(self.relu4(self.conv4(x)))

        # x = x.view(-1, self.flattened_size)

        # x = self.fc(x)
        x= self.model(x)
        return x

import torch

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}  # Stores the EMA parameters
        self.original = {} # Stores the original parameters temporarily

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        # Temporarily replace model parameters with EMA parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        # Restore original model parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.original:
                param.data = self.original[name]
        self.original = {}

def train_model(model, train_loader, criterion, optimizer, device, ema):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        ema.update(model)

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct_predictions = (predicted == labels).sum().item()
        total_samples = labels.size(0)
        accuracy = (correct_predictions / total_samples) * 100
        running_accuracy += accuracy * images.shape[0]

        # if idx % 10 == 1 :
        #     print(accuracy, loss)
        

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = running_accuracy / len(train_loader.dataset)
    return epoch_loss, epoch_accuracy


wandb.login(key='c85c2ca4b05cbf60cbd4154f1c420aae8dce9c68')
wandb.init(project='MLNS-1')

input_channels = 1  
num_classes = 37
learning_rate = 0.001
batch_size = 256
num_epochs = 150


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNet(input_channels=input_channels, num_classes=num_classes).to(device)
ema = EMA(model, decay=0.99)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_data0 = np.concatenate([np.load('data0.npy'), np.load('data1.npy')], axis=0)
train_lab0 = np.concatenate([np.load('lab0.npy'), np.load('lab1.npy')], axis=0)

train_data1 = np.load('data2.npy')
train_lab1 = np.load('lab2.npy')


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel 3 times
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

train_loader = torch.utils.data.DataLoader(MyData(train_data0, train_lab0, transform), batch_size=batch_size, num_workers=7)
val_loader = torch.utils.data.DataLoader(MyData(train_data1, train_lab1, transform), batch_size=batch_size, num_workers=7)


# Training loop
print('Start training')
for epoch in range(num_epochs):
    model.train()
    ema.restore(model)
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device, ema)

    ema.apply_shadow(model)
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions = (predicted == labels).sum().item()
            total_samples = labels.size(0)
            val_accuracy += (correct_predictions / total_samples) * 100 * images.size(0)
    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Validation Loss: {val_loss}')
    print(f'Train accuracy: {train_accuracy}, Validation accuracy: {val_accuracy}' )
    wandb.log(
        {
            'Epoch':epoch+1,
            'Train Loss':   train_loss,
            'Validation Loss': val_loss,
            'Train accuracy': train_accuracy,
            'Validation accuracy': val_accuracy,

        }
    )
    torch.save(model, 'resnet50-ema-model-epoch.pth')

torch.save(model, 'resnet50-ema-model.pth')