
import pdb
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn import *
from sklearn.metrics import accuracy_score
from sklearn import metrics

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"



transform = transforms.Compose(
    [transforms.Resize((224,224)),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
writer=SummaryWriter(r"C:\runs\transformation_non_horizontal_validation_18")


### CHANGE ###
data_path = r"D:\melanoma pictures\train_sep"

trainset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transform
)
##############
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=0)

### CHANGE ###
data_path = r"D:\melanoma test\test"
testset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transform
)
##############
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=True, num_workers=0)

classes = ('Melanoma', 'NotMelanoma')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
print (labels)
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120) ## CHANGE
        self.fc2 = nn.Linear(120, 2)
        self.fc3 = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53) ## CHANGE
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(60):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        writer.add_scalar('training loss', loss, epoch*len(trainloader)+i)
        y_true=labels.numpy()
        y_pred=torch.argmax(outputs,dim=1).detach().numpy()
        try:
            f1_score=metrics.f1_score(y_true, y_pred)
        except:
            tn,fp,fn,tp=metrics.confusion_matrix(y_true,y_pred).ravel()
            precision=tp/(tp+fn)
            recall=tp/(tp+fp)
            f1_score=precision*recall/((precision+recall)*2)
            
        writer.add_scalar('training f1_score', f1_score, epoch*len(trainloader)+i)
        ac_score=accuracy_score(y_true,y_pred,normalize=True)
        writer.add_scalar('training ac_score',ac_score,epoch*len(trainloader)+i)


        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    
            dataiter = iter(testloader)
            inputs, labels = dataiter.next()
            for i, data in enumerate(testloader, 0):
                optimizer.zero_grad()
                outputs=net(inputs)
                loss=criterion(outputs,labels)
                writer.add_scalar('validation loss', loss, epoch*len(trainloader)+i)
                optimizer.zero_grad()
                y_true=labels.numpy()
                y_pred=torch.argmax(outputs,dim=1).detach().numpy()
                f1_score=metrics.f1_score(y_true,y_pred)
                writer.add_scalar('validation f1_score', f1_score, epoch*len(trainloader)+i)
                ac_score=accuracy_score(y_true,y_pred,normalize=True)
                writer.add_scalar('validation ac_score',ac_score,epoch*len(trainloader)+i)
        






    

print('Finished Training')
writer.close()
