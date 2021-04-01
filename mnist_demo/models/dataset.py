# SageMaker paths
import torch

class MyMNIST(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=None, target_transform=None):
        self.transform = transform
        # Loading local MNIST files in PyTorch format: training.pt and test.pt.
        self.data, self.labels = torch.load(path)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)
