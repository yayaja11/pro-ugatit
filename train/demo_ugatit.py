#-- coding:UTF-8 --

import sys
import os
import torch
import cv2
from PIL import Image
from torchvision.utils import save_image, make_grid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms
from models.ugatit_pro import UGATIT
from configs.cfgs_ugatit import test_cfgs as ugatit_cfgs
# from configs.cfgs_ugatit import cfgs as ugatit_cfgs

from utils.utils import load_image, check_dir_existing
import numpy as np
from glob import glob
import argparse


def preprocessing(img):
    # Convert the image to a PyTorch tensor and normalize the pixel values to the range [0, 1].
    img = torch.from_numpy(img).float() / 255.0

    # Permute the dimensions to match the expected input shape for the model (channels-first).
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img

# Disable gradient computation to speed up inference.
torch.set_grad_enabled(False)

# Comment out the unused import to avoid confusion.
# face_align_lib = import_module('3rdparty.face_alignment.api')

def read_img_path(path, s=512):
    # Read an image using PIL and convert it to RGB.
    img = Image.open(path).convert('RGB')

    # Resize the image to a square of size s x s.
    img = img.resize(size=(s, s))

    # Define a sequence of transformations to apply to the image.
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor.
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the pixel values.
    ])
    # Apply the transformations and add a batch dimension to the tensor.
    img_PIL_Tensor = transform1(img).unsqueeze(0)
    return img_PIL_Tensor

if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default="ugatit", type=str)
    parser.add_argument('--resume', default="weights/anime/train_latest.pt", type=str)
    parser.add_argument('--input', default="test_inputs", type=str)
    parser.add_argument('--saved-dir', default="test_outputs", type=str)
    parser.add_argument('--dataset', default="gyate", type=str)
    parser.add_argument('--align', action='store_true', default=False)
    parser.add_argument('--anime', action='store_true', default=False)
    args = parser.parse_args()

    # Extract the model type, image paths, weight location, and output directory from the arguments.
    model_type = args.type
    images_path = args.input
    weight_loc = args.resume
    saved_dir = args.saved_dir

    # Create the output directory if it doesn't exist.
    saved_dir = os.path.join(saved_dir, model_type)
    check_dir_existing(saved_dir)

    # Set the size of the input images.
    size = 512

    # Get a list of image paths in the input directory.
    images_set = glob(images_path + '/*')

    # Load the specified model and set its mode to evaluation.
    if model_type == 'ugatit':
        ugatit_cfgs.anime = False
        model = UGATIT(ugatit_cfgs)
    else:
        raise ValueError('model type error.')
    # G_A = model.G_A
    # G_B = model.G_B

    # Load the weights for the model.
    import os
    pathLoc = os.path.join("results", "preview","model", "train_20.pt")
    weight_set = torch.load(pathLoc, map_location='cuda')
    # for name, param in weight_set.items():
    #     import pdb;pdb.set_trace()
        # for param_name, param in param.named_parameters():
            
            # print(param_name,len(param))
    
    model.G_A.load_state_dict(weight_set['G_A'])
    model.G_A.eval()
    model.G_B.load_state_dict(weight_set['G_B'])
    model.G_B.eval()

    # Check if a GPU is available for inference.
    if torch.cuda.is_available():
        model.G_A.cuda()
        model.G_B.cuda()
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    # Get a list of file names from the input directory.
    names = os.listdir("testdata/character")
    count = -1
    # Process each image in the input directory.
    for image_path in images_set:
        count += 1
        name = names[count]
        print('starting to transform {}'.format(image_path))

        # Load and preprocess the input image.
        img_tensor = read_img_path(image_path, size).to(dev)

        # Generate fake images using the model in both directions (A to B and B to A).
        realA = make_grid(img_tensor, padding=2, normalize=True, value_range=(-1, 1))
        fakeB = model.G_B.test_forward(img_tensor, 'BtoA')
        fakeB = make_grid(fakeB, padding=2, normalize=True, value_range=(-1, 1))

        fakeC = model.G_A.test_forward(img_tensor, 'AtoB')
        fakeC = make_grid(fakeC, padding=2, normalize=True, value_range=(-1, 1))

        # Save the output image grid to the specified directory.
        savePath = os.path.join(saved_dir, '{}'.format(name))
        image_grid = torch.cat((realA, fakeB, fakeC), 1)
        
        try:
            save_image(image_grid, savePath, normalize=True)
        except Exception as e:
            print(e)

    print('pred done, saving images to "{}" dir.'.format(saved_dir))
