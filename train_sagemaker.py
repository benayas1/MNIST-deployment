from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from mnist_demo.models.model import Net
from mnist_demo.models.dataset import MyMNIST
import os
import ssl

from sagemaker_inference import (
    content_types,
    decoder,
    encoder,
    errors,
    utils,
)

ssl._create_default_https_context = ssl._create_unverified_context


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 250 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def model_fn(model_dir):
    """
    Function used for Sagemaker to load a model. The function must have this signature. Sagemaker will look for this function.
    Used only when Elastic Inference is not used.
    """
    print('Loading model')
    model = Net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f: # model_cnn.pth is the name given in the train script
        model.load_state_dict(torch.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) #let's keep inference in CPU
    print('Model loaded')
    return model


def input_fn(request_body, request_content_type):
    """
    This function receives a payload and a content type and produces a python object.
    The standard input_fn provided by the framework is able to open NPY, JSON, CSV and NPZ objects, and formats them into 
    numpy arrays.
    This function will format data into Torch tensors.
    More info in https://github.com/aws/sagemaker-inference-toolkit/tree/master/src/sagemaker_inference"""
    print('input function')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np_array = decoder.decode(request_body, request_content_type) # this function figures out the content type and returns numpy array
    tensor = torch.FloatTensor(np_array) if request_content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    return tensor.to(device)

def predict_fn(input_data, model):
    """
    The model object is the model returned from model_fn.
    The input_data is the output of input_fn
    """
    print('predict function')
    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                                 ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(transform(input_data.to(device)))

def output_fn(prediction, content_type):
    """
    This function formats the prediction, which in this case is a Torch tensor, into a type defined by content_type
    """
    print('output function')
    if type(prediction) == torch.Tensor:
            prediction = prediction.detach().cpu().numpy().tolist()

    for content_type in utils.parse_accept(content_type):
        if content_type in encoder.SUPPORTED_CONTENT_TYPES:
            encoded_prediction = encoder.encode(prediction, content_type)
            if content_type == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")
            return encoded_prediction

    raise errors.UnsupportedFormatError(content_type)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',  help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=7, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--use_cuda', type=bool, default=False)

    print(f"Source {os.environ['SM_MODEL_DIR']}")
    print(f"Channel {os.environ['SM_CHANNEL_TRAINING']}")
    # Data, model, and output directories
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--channel', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    #parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    model_dir = args.model_dir

    use_cuda = args.use_cuda and torch.cuda.is_available()

    torch.manual_seed(42)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device is {}'.format(device))

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    print(f'Downloading dataset from {args.channel}')
    dataset1 = MyMNIST(os.path.join(args.channel, 'training.pt'), transform=transform)
    dataset2 = MyMNIST(os.path.join(args.channel, 'test.pt'), transform=transform)

    print('Dataset downloaded successfully')
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    savepath = os.path.join(model_dir, 'model.pth')
    with open(savepath, 'wb') as f:
        print(f'Saving model into {savepath}')
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    main()