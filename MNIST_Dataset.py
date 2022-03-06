import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


trainDataset = MNIST(root="D:\\WorkSpace\\DataSet", train=True, transform=torchvision.transforms.ToTensor())
testDataset = MNIST(root="D:\\WorkSpace\\DataSet", train=False, transform=torchvision.transforms.ToTensor())
batchSize = 100
trainSize = trainDataset.__len__()
testSize = testDataset.__len__()
trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=True)
testLoader = DataLoader(testDataset, batch_size = batchSize, shuffle=False)
