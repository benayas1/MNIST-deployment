# This file implements functions model_fn, input_fn, predict_fn and output_fn.
# Function model_fn is mandatory. The other functions can be omitted so the standard sagemaker function will be used.
# An alternative to the last 3 functions is to use function transform_fn(model, data, input_content_type, output_content_type)
#
# More info on https://github.com/aws/sagemaker-inference-toolkit/tree/master/src/sagemaker_inference
#

import torch
from mnist_demo.models.model import Net
import os
import io
from torchvision import transforms
from sagemaker_inference import (
    content_types,
    decoder,
    encoder,
    errors,
    utils,
)

def model_fn(model_dir):
    """
    Function used for Sagemaker to load a model. The function must have this signature. Sagemaker will look for this function.
    Used only when Elastic Inference is not used.
    """
    model = Net()
    with open(os.path.join(model_dir, 'model_cnn.pth'), 'rb') as f: # model_cnn.pth is the name given in the train script
        model.load_state_dict(torch.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) #let's keep inference in CPU
    return model

def input_fn(request_body, request_content_type):
    """
    This function receives a payload and a content type and produces a python object.
    The standard input_fn provided by the framework is able to open NPY, JSON, CSV and NPZ objects, and formats them into 
    numpy arrays.
    This function will format data into Torch tensors.
    More info in https://github.com/aws/sagemaker-inference-toolkit/tree/master/src/sagemaker_inference"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np_array = decoder.decode(request_body, request_content_type) # this function figures out the content type and returns numpy array
    tensor = torch.FloatTensor(np_array) if request_content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
    return tensor.to(device)

def predict_fn(input_data, model):
    """
    The model object is the model returned from model_fn.
    The input_data is the output of input_fn
    """
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
    if type(prediction) == torch.Tensor:
            prediction = prediction.detach().cpu().numpy().tolist()

    for content_type in utils.parse_accept(content_type):
        if content_type in encoder.SUPPORTED_CONTENT_TYPES:
            encoded_prediction = encoder.encode(prediction, content_type)
            if content_type == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")
            return encoded_prediction

    raise errors.UnsupportedFormatError(content_type)
