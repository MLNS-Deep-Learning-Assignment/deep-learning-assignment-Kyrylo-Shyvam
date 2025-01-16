import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

import torchvision
import itertools

from torchvision import models
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


import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Pad(padding=(0, 8,0,8)),
    #  transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomResizedCrop((40, 41), scale=(0.7, 0.9), ratio=(0.9, 1.1)),
])

transform_test = transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.060661207217261905,),(0.2193603594906726)),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13015008378839818,), (0.30690425774764835,)),
    # transforms.Pad(padding=(0, 8,0,8)),
     transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomResizedCrop((40, 20), scale=(0.7, 0.9), ratio=(0.9, 1.1)),   
])

trainset = torchvision.datasets.MNIST(
    root='./data',  # Directory where data will be stored
    train=True,     # Get the training set
    download=True,  # Download the dataset if it's not already present
    transform=transform  # Apply the defined transformations
)

testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,  # Get the test set
    download=True,
    transform=transform)
    
class SyntheticData(torch.utils.data.Dataset):
    def __init__(self, data, label, transform = transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        length = 10000
        index = torch.randint(0, length, (4,),)
        # print(index, length)
        final_img = []
        final_label = []

        for i in range(4):
            # print(self.data[index[i]].shape,)
            final_img.append(self.transform(self.data[index[i]]))
            final_label.append(self.label[index[i]])
            # print(final_img[-1].shape)
        

        return torch.cat(final_img, -1), torch.stack(final_label, -1)

    def __len__(self):
        return 10000


class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # Assuming input images are 40 x 168

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc= nn.ModuleList([ nn.Linear(self.num_ftrs, 10) for _ in range(4)])
        self.model.fc.requires_grad_(True)
        # self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):

        x= self.model(x)
        y = [ ]
        for _ in range(4):
            y.append(self.fc[_](x))
    
        return torch.stack(y, dim=-1)

import tqdm
def train_model(model, train_loader, criterion, optimizer, device, ema):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for idx, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        # print(images.device, labels.device)
        optimizer.zero_grad()
        outputs = model(images)

        # print(outputs.shape, labels.shape)
        # exit()
        # print(outputs[0], labels[0])
        loss = criterion(outputs, labels)
        # print(loss)
        loss.backward()
        optimizer.step()

        ema.update(model)


        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        # print(predicted[0], labels[0])
        correct_predictions = (predicted == labels).sum().item()/4
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
num_classes = 10
learning_rate = 0.0001
batch_size = 128*5
num_epochs = 150


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model = ConvNet(input_channels=input_channels, num_classes=num_classes).to(device)


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
ema = EMA(model, decay=0.99)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_data0 = np.load('data0.npy')
train_lab0 = np.load('lab0.npy')

train_data1 = np.load('data1.npy')
train_lab1 = np.load('lab1.npy')


class CustomLoss():
    def __init__(self):
        numbers = list(range(10))  # Represents numbers 0 to 9
        all_sets = torch.tensor(list(itertools.product(numbers, repeat=4))).detach()
        self.all_sums = [[] for _ in range(37)]
        for vector in all_sets:
            sm = vector.sum()
            self.all_sums[sm].append(vector)

        for i in range(37):
            # print(i, self.all_sums[0].shape)
            self.all_sums[i] = torch.stack(self.all_sums[i], dim=0)

    def calculate(self, logits, all_sum):
        # logits are b x 10 x 4
        probability = logits.softmax(dim=1)
        loss = 0.0
        # print(self.all_sums)
        # construct b x number
        for logit,one_sum  in zip(probability, all_sum):
            # print(logit,one_sum)
            all_prob = logit[self.all_sums[one_sum], torch.arange(4)]
            loss += -torch.log(all_prob.prod(-1).sum())
            # for vector4 in self.all_sums[one_sum]:

            #     x = logit[vector4, torch.arange(4)].prod(-1)
            #     print(logit[vector4, torch.arange(4)], vector4, logit[vector4, torch.arange(4)].prod(-1))
            #     loss += x

            # print(loss)
        loss = loss/all_sum.shape[0]
        return loss
    

class SyntheticDataset10000(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[idx]

    def __len__(self):
        return len(self.targets)

ds = torch.load('./dataset50k.pth')
ds2 = torch.load('./dataset100.pth')
transform_synth = transforms.Compose([
    transforms.Normalize((0.0392,), (0.1543,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # transforms.RandomResizedCrop((40, 164), scale=(0.8, 0.9), ratio=(0.9, 1.1)),
    # transforms.Resize((40, 164))
])
train_synth = SyntheticDataset10000(ds.data, ds.targets, transform=transform_synth)
val_synth = SyntheticDataset10000(ds2.data, ds2.targets, transform=transform_synth)


# train_synth = SyntheticData(trainset.data.numpy(), trainset.targets, transform)
# test_synth = SyntheticData(testset.data.numpy(), testset.targets, transform)

train_loader = torch.utils.data.DataLoader(train_synth, batch_size=batch_size, num_workers=5)
val_loader = torch.utils.data.DataLoader(val_synth, batch_size=batch_size, num_workers=5)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.060661207217261905,),(0.2193603594906726,)),
])

# train_loader = torch.utils.data.DataLoader(MyData(train_data0, train_lab0, transform=transform_test), batch_size=batch_size, num_workers=5)
test_loader = torch.utils.data.DataLoader(MyData(train_data1, train_lab1, transform=transform_test), batch_size=batch_size, num_workers=5)

# ls = CustomLoss()
# criterion = ls.calculate

import torchvision.transforms as transforms

def validation(val_loader, model, criterion, device):
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm.tqdm(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # print(outputs.shape, labels.shape)

            loss = criterion(outputs, labels)
            # print(loss)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct_predictions = (predicted == labels).sum().item()/4
            total_samples = labels.size(0)
            val_accuracy += (correct_predictions / total_samples) * 100 * images.size(0)
    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)

    return val_loss, val_accuracy

ls  = CustomLoss()
criterion2 = ls.calculate
def special_validation(val_loader, model, criterion, device):
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm.tqdm(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probab = outputs.softmax(-2).max(-2)[0].prod(-1)
            # print(probab[:2], outputs.softmax(-2)[:2], outputs.softmax(-2).max(-2)[0][:2])
            # print(-torch.log(probab).mean(), probab[:10])
            # exit()
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(-2).sum(-1)
            correct_predictions = (predicted == labels).sum().item()
            total_samples = labels.size(0)
            val_accuracy += (correct_predictions / total_samples) * 100 * images.size(0)
    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)

    return val_loss, val_accuracy

# Training loop
import tqdm
print('Start training')
for epoch in range(num_epochs):
    # train_loss, train_accuracy  =
    model.train()
    ema.restore(model)
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device, ema)

    ema.apply_shadow(model)
    model.eval()
    val_loss, val_accuracy = validation(val_loader, model, criterion, device)
    test_loss, test_accuracy =  special_validation(test_loader, model, criterion2, device)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Validation Loss: {val_loss}, Test loss: {test_loss} ', )
    print(f'Train accuracy: {train_accuracy}, Validation accuracy: {val_accuracy}, Test Accuracy: {test_accuracy}' )
    wandb.log(
        {
            'Epoch':epoch+1,
            'Train Loss':   train_loss,
            'Validation Loss': val_loss,
            'Train accuracy': train_accuracy,
            'Validation accuracy': val_accuracy,

        }
    )
    torch.save(model, 'synthetic-ema-model-epoch.pth')

torch.save(model, 'synthetic-ema-model.pth')