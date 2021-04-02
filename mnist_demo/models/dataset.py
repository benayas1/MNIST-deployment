# SageMaker paths
import torch
from PIL import Image

class MyMNIST(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.transform = transform
        # Loading local MNIST files in PyTorch format: training.pt and test.pt.
        self.data, self.labels = torch.load(path)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
