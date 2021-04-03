# This file implements functions model_fn, input_fn, predict_fn and output_fn.
# Function model_fn is mandatory. The other functions can be omitted so the standard sagemaker function will be used.
# An alternative to the last 3 functions is to use function transform_fn(model, data, input_content_type, output_content_type)
#
# More info on https://github.com/aws/sagemaker-inference-toolkit/tree/master/src/sagemaker_inference
#

import torch
from mnist_demo.models.model import Net
import os
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
    print('Loading model')
    model = Net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f: # model_cnn.pth is the name given in the train script
        model.load_state_dict(torch.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) #let's keep inference in CPU
    print('Model loaded')
    return model