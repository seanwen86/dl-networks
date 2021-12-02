'''
Sean Wen
'''

from torch.utils import data
from torchvision.transforms.transforms import Lambda
from AlexNet import AlexNetOriginal, AlexNet2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, dataset

from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TrainConfig = {}

RANDOM_SEED = 42
LEARNING_RATE = 0.1
BATCH_SIZE = 32
N_EPOCHS = 20

IMG_SIZE = 32
N_CLASSES = 10

# class ToColorfulImage():
#     def __call__(self, img):
#         return img.repeat(3, 1, 1)
# transforms = transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor(), ToColorfulImage()])
# train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms, download=True)
# test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms, download=True)
# labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# https://www.cs.toronto.edu/~kriz/cifar.html
transforms = transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms, download=True)
labels = ['airplane', 'automobile', 'bird', 'Cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


ROW_IMG = 10
N_ROWS = 5
fig = plt.figure()
for index in range(1, ROW_IMG*N_ROWS+1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(train_dataset.data[index])
fig.suptitle('CIFAR10 Dataset-preview')
plt.show()

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

torch.manual_seed(RANDOM_SEED)
net = AlexNet2(n_classes = N_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

def train_loop(net, train_loader, criterion, optimizer, device):
    net.train()

    running_loss = 0
    for x, y_true in train_loader:
        optimizer.zero_grad()

        x = x.to(device)
        y_true = y_true.to(device)

        y_hat, _ = net(x)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item()*x.size(0)

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss

def test_loop(net, test_loader, criterion, device):

    net.eval()
    running_loss = 0

    for x, y_true in test_loader:

        x = x.to(device)
        y_true = y_true.to(device)

        y_hat, _ = net(x)
        loss = criterion(y_hat, y_true)

        running_loss += loss.item()*x.size(0)
    
    test_loss = running_loss / len(test_loader.dataset)

    return test_loss

def get_accuracy(net, data_loader, device):

    net.eval()
    num_correct = 0

    for x, y_true in data_loader:
        x = x.to(device)
        y_true = y_true.to(device)

        _, probs = net(x)
        y_pred = torch.argmax(probs, dim=1)
        # print(y_pred)

        num_correct += (y_true == y_pred).sum()

    accuracy = num_correct / len(data_loader.dataset)
    return accuracy

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')

def train(net, criterion, optimizer, train_loader, test_loader, epochs, device):

    best_loss = 1e10
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss = train_loop(net, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        with torch.no_grad():
            test_loss = test_loop(net, test_loader, criterion, device)
            test_losses.append(test_loss)

            train_acc = get_accuracy(net, train_loader, device)
            test_acc = get_accuracy(net, test_loader, device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Test loss: {test_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Test accuracy: {100 * test_acc:.2f}')
        
    plot_losses(train_losses, test_losses)

train(net, criterion, optimizer, train_loader, test_loader, N_EPOCHS, device)

ROW_IMG = 10
N_ROWS = 5
fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(test_dataset.data[index], cmap='gray_r')
    
    with torch.no_grad():
        net.eval()
        x, _ = test_dataset[index]
        x = x.unsqueeze(0).to(device)
        _, probs = net(x)
        
    title = f'{labels[torch.argmax(probs)]} ({torch.max(probs * 100):.0f}%)'
    
    plt.title(title, fontsize=7)
fig.suptitle('AlexNet - predictions')
plt.show()