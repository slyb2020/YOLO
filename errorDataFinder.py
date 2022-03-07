import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from LeNet import LeNet5
from LeeNet import LeeNet
import time
from MNIST_Dataset import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from ID_DEFINE import *


def ErrorTest(structure,preModel) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # leNet5 = LeNet5()
    if structure == 0:
        model = LeNet5()
    elif structure == 1:
        model = LeeNet()
    model = torch.load('./model/' + LeNetModelList[structure] + '/' + preModel)
    model.to(device)
    model.eval()
    accuracyTotal = 0
    batch = 0
    errorList = []
    with torch.no_grad():
        for imgs, labels in testLoader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            predicts = model(imgs)
            accuracy = (torch.argmax(predicts, 1) == labels).sum()
            a = np.array(torch.Tensor.cpu(torch.argmax(predicts, 1) == labels))
            for i, result in enumerate(a):
                if not result:
                    errorList.append([i + batch * batchSize, torch.Tensor.cpu(labels[i]).detach().item(), torch.argmax(predicts, 1)[i].item()])
            # print(a)
            accuracyTotal += accuracy
            batch += 1
    return errorList

if __name__ == "__main__":
    a = ErrorTest()
    print(len(a), a)
# errorList.append(0)
# errorList.append(0)
# print(errorList)
# print(len(errorList))
# print(accuracyTotal.item())
# # writer.add_images("pic",img)
# error = np.array(errorList)
# error.reshape((-1,10))
# for row in range(6):
#     for col in range(10):
#         img, label = testDataset[error[row*10+col]]
#         print(error[row*10+col],label)
#         writer.add_image("row%d"%row, img, col)
# writer.close()
