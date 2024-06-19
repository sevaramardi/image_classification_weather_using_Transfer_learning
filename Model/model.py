import torch
import torchvision
from torchvision import models
import torch.nn as nn
import numpy as np
import time
import os
from pre_processing import test_loader,train_loader


losses = []
accs = []
def train_model(model, criterion, optimizer, schedular, num_epochs=5):
    since = time.time()

    for epoch in range(num_epochs):
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch}/{num_epochs-1}")
        print('-'*40)
        running_loss = 0.0
        running_correct = 0

        for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device).long()
                optimizer.zero_grad()
                output = model(images)
                _,preds = torch.max(output,1)
                loss = criterion(output,labels)
                loss.backward()
                optimizer.step()

                running_loss +=loss.item()*images.size(0)
                running_correct  += torch.sum(preds == labels.data)
                schedular.step()
        
        epoch_loss = running_loss/len(train_loader)
        epoch_accc = running_correct/ len(train_loader)
        losses.append(epoch_loss)
        accs.append(epoch_accc)
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_accc:.4f}')

def vizualization(lst_losses, lst_accs, num_epochs):
     plt.plot(num_epochs, lst_losses)
     plt.xlabel('Epoch')
     plt.ylabel('Loss')
     plt.title('Train Loss')
     plt.plot(num_epochs, lst_accs)
     plt.xlabel('Epoch')
     plt.ylabel('Accuracy')
     plt.title('Train Accuracy')

     return plt.show()



model = models.resnet18(weights=True)
num_ftrs = model.fc.in_features
models.fc = nn.Linear(num_ftrs, 11)

model = model.to(device)
creterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

step_schedular = lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)
model = train_model(model, creterion, optim, step_schedular)
print(vizualization(losses,accs,5))


