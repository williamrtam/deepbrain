import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

def CreateXYArrays():
    '''
    reads in data partially pre-processed in matlab and applies Sobel filter.
    reshapes to 28x28 numpy arrays for training data

    '''
    X = []
    y = []

    folder = 'True'
    filenames = [f for f in os.listdir(folder) if f.endswith('.png')]
    for x in range(len(filenames)):
        im = cv2.imread(folder + '/' + filenames[x], cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (28,28), interpolation = cv2.INTER_AREA)
        im = cv2.Sobel(im,cv2.CV_8U,1,1,ksize=5)
        im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        if im.shape[0] == 28:
            X.append(im)
            y.append(1.0)

    folder = 'False'
    filenames = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    for x in range(len(filenames)):
        im = cv2.imread(folder + '/' + filenames[x], cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (28,28), interpolation = cv2.INTER_AREA)
        im = cv2.Sobel(im,cv2.CV_8U,1,1,ksize=5)
        im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        if im.shape[0] == 28:
            X.append(im)
            y.append(0.0)

    X = np.array(X)
    y = np.array(y)

    np.save('Xdata.npy', X)
    np.save('Ydata.npy', y)

    return


class BrainDataset(Dataset):
    def __init__(self):
        X = np.load('Xdata.npy')
        y = np.load('Ydata.npy')
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).unsqueeze(1).float()
        self.y_data = torch.from_numpy(y).unsqueeze(1).float()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class BrainConvolutionalNN(nn.Module):
    def __init__(self):
        super(BrainConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, 1, 0)
        self.conv2 = nn.Conv2d(16, 8, 7, 1, 0)
        self.fc1 = nn.Linear(16*16*8, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1) 
        out = F.sigmoid(self.fc1(out))

        return out

def BrainConvNN():
    net = BrainConvolutionalNN()
    # adds in gpu support
    net.cuda()

    # X_train,y_train = extract_data(, )
    # X_train,y_train = extract_data('datasets/random/random_imgs.npy', 'datasets/random/random_labs.npy')
      
    dataset = BrainDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=100,
                              shuffle=True)

    # Specify the loss function
    criterion = nn.BCELoss()

    # Specify the optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0, weight_decay=0)

    #RUN FOR 10000 EPOCHS
    max_epochs = 100

    loss_np = np.zeros((max_epochs))
    trainaccuracy = np.zeros((max_epochs))
    trainrecall = np.zeros((max_epochs))
    trainprecision = np.zeros((max_epochs))
    trainf1 = np.zeros((max_epochs))


    for epoch in range(max_epochs):
        traincorrect = 0
        losssum = 0
        traintotal = 0
        testtotal = 0
        truepositive = 0
        falsepositive = 0
        falsenegative = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            traintotal += len(labels)
            y_pred = net(inputs)
            loss = criterion(y_pred, labels)
            
            print("epoch: ", epoch, "loss: ", loss.data[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_np = y_pred.data.cpu().numpy()
            
            # calculate the training accuracy of the current model
            label_np = labels.data.cpu().numpy().reshape(len(labels),1)
            pred_np = np.where(y_pred_np>0.5, 1, 0) 
            for j in range(y_pred_np.shape[0]):
                if pred_np[j] == label_np[j]:
                    traincorrect += 1
                if pred_np[j] == 1:
                    if label_np[j] == 1:
                        truepositive += 1
                    else:
                        falsepositive += 1
                else:
                    if label_np[j] == 1:
                        falsenegative += 1

            losssum += loss.data.cpu().numpy()

        trainaccuracy[epoch] = float(traincorrect)/float(traintotal)
        trainrecall[epoch] = float(truepositive)/float(truepositive + falsepositive)
        trainprecision[epoch] = float(truepositive)/float(truepositive + falsenegative)
        trainf1[epoch] = 2*(trainprecision[epoch]*trainrecall[epoch])/(trainprecision[epoch]+trainrecall[epoch])       
        loss_np[epoch] = float(losssum) / float(traintotal)



    print("final training accuracy: ", trainaccuracy[epoch])
    print("final training recall: ", trainrecall[epoch])
    print("final training precision: ", trainprecision[epoch])
    print("final training f1: ", trainf1[epoch])


    epoch_number = np.arange(0,epoch+1,1)

    # Plot the loss over epoch
    plt.figure()
    plt.plot(epoch_number, loss_np[0:epoch+1])
    plt.title('loss over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')

    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, trainaccuracy[0:epoch+1])
    plt.title('training accuracy over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')


    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, trainrecall[0:epoch+1])
    plt.title('training recall over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')


    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, trainprecision[0:epoch+1])
    plt.title('training precision over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')


    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, trainf1[0:epoch+1])
    plt.title('training f1 over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    plt.show()

# CreateXYArrays()
BrainConvNN()