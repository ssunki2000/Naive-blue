#coding: utf-8
#1.import packages
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets


#2.Def Hyperparameters
batch_size = 100
learning_rate = 0.05
num_epoches = 50



#3.import MNIST

data_tf = torchvision.transforms.Compose(                                                              #pretreat the data
    [
        torchvision.transforms.ToTensor(),                                                             #（0，1）
        torchvision.transforms.Normalize([0.5],[0.5])                                                  #（-1，1）
    ]
)



data_path = r'C:\Users\ssunki\Anaconda3\MNIST'                                                         #import dataset

train_dataset = datasets.MNIST(data_path, train=True,  transform=data_tf, download=False)
test_dataset  = datasets.MNIST(data_path, train=False, transform=data_tf, download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)            #import iteration data
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=0)








#4.Def net Model

class Cnn(nn.Module):
    def __init__(self):                                                                               #dim,classify
        super(Cnn, self).__init__()                                                                   # Cnn inherit nn.Module
        self.conv = nn.Sequential(
            nn.Conv2d(1,  6, 5, 1, 2),                                                                #parameters:in_dim，out_dim，ksize，stride，padding)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(                                                                      #FC
            nn.Linear(16*5*5, 120),
            nn.Linear(120*1*1, 84),
            nn.Linear(84*1*1, 10))

    def forward(self, x):                                                                             #forward
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = Cnn()
print(model)
model = model.cuda()                                                                                  #call gpu



# 5.Def loss func and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)




# 6.train
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))

    running_loss = 0.0                                                                               #initialize loss & acc
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.cuda()
        label = label.cuda()
        img = Variable(img)                                                                          #transforms tensor into variable
        label = Variable(label)

                                                                                                     # forward
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()

                                                                                                     # back forward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))



#7.Def test_model


    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        # Wrap in variable
        with torch.no_grad():
            img = Variable(img).cuda()
            label = Variable(label).cuda()

        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()


#8.Print loss & acc

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
        eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
    print()

import winsound                                                                                      #ring

winsound.Beep(32767,2000)

'''
#打印一张示例图
print(train_dataset.train_dataset.size())                                                            # (60000, 28)
print(train_dataset.train_labels.size())                                                             # (60000)
plt.imshow(train_dataset.train_dataset[5].numpy(), cmap='gray')
plt.title('%i' % train_dataset.train_labels[5])
plt.show()
'''
