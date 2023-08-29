'''
DISCARDED
This test is integrated to the main pipeline of this project
'''''
import matplotlib
matplotlib.use('TkAgg')
import os
import torch
import torchvision as torchvision
from torch.utils.data import DataLoader
from AutoCrop.auto_crop_model_utility import save_out_label
from AutoCrop.utility import getName
from AutoCrop.auto_crop_Dataset import TestMain
from AutoCrop.auto_crop_model_UNet import UNet



def UNet_pipeline(model_path, val_input_dir, val_output_dir):
    '''
    Main function for auto_crop testing
    '''''
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))   # load the pre-trained mode
    model.to(device)  # Move the model to the appropriate device (CPU or GPU)

    # Load data
    val_dataset = TestMain(val_input_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Retrieve the name of files for saving
    name_list = getName(val_input_dir)

    i = 0   # used to name the file
    for image in val_loader:
        image = image.to(device)
        image.requires_grad = True
        model.eval()
        # the code within torch.no_grad() will not track the gradient
        with torch.no_grad():
            out = model(image)
            # visualize and save the output mask
            save_out_label(out, i, name_list, val_output_dir)
            i += 1


