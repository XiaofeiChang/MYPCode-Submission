'''
This test is used to test the performance of UNet
'''''
import matplotlib
matplotlib.use('TkAgg')
import torch
from torch.utils.data import DataLoader
from AutoCrop.auto_crop_model_utility import *
from AutoCrop.auto_crop_Dataset import TrainValDataset
from AutoCrop.auto_crop_model_UNet import UNet
from AutoCrop.utility import getName



def main():
    '''
    Main function for auto_crop testing
    '''''
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = torch.load('./auto_crop_unet.pth', map_location=device)   # load the pre-trained mode
    model = UNet(n_channels=1, n_classes=1, bilinear=False)
    model.load_state_dict(torch.load('./auto_crop_unet.pth', map_location=device))
    model.to(device)  # Move the model to the appropriate device (CPU or GPU)
    # Load data
    _, _, val_input_dir, val_label_dir, val_output_dir = getPath()
    val_dataset = TrainValDataset(val_input_dir, val_label_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Retrieve the name of files for saving
    name_list = getName(val_input_dir)

    i = 0  # used to name the file

    # Test images one by one
    for image, label in val_loader:
        image = image.to(device)
        image.requires_grad = True

        model.eval()
        # the code within torch.no_grad() will not track the gradient
        with torch.no_grad():
            out = model(image)
            # visualize and save the output mask
            save_out_label(out, i, name_list, val_output_dir)
            i += 1

if __name__ == '__main__':
    main()