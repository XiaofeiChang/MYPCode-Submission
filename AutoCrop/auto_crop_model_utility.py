'''
This file includes the loss functions or accuracy functions of UNet, as well as the plotting functions.
'''''
import os
from tifffile import imread, imsave  # Fiji
from torchvision import transforms
import numpy as np
import sys
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)  # used to print full numpy array



def getPath():
    '''
    Get the path of the DATASET directory.
                MYPCode
            /        \
        Preprocessing Alignment
    '''''
    proj_dir = 'drive/MyDrive/Y4T3____MYPCode/Preprocessing/'
    train_input_dir = proj_dir + '/DATASET' + '/Preprocessing' + '/input_train/'
    train_label_dir = proj_dir + '/DATASET' + '/Preprocessing' + '/label_train/'
    val_input_dir = proj_dir + '/DATASET' + '/Preprocessing' + '/input_val/'
    val_label_dir = proj_dir + '/DATASET' + '/Preprocessing' + '/label_val/'
    val_output_dir = proj_dir + '/DATASET' + '/Preprocessing' + '/output_val/'
    return train_input_dir, train_label_dir, val_input_dir, val_label_dir, val_output_dir


def IoU(output, label):
    '''
    Calculate Intersection over Union (IoU) Accuracy. IoU = TP / (TP + FP + FN)
    :param output: 2d tensor: output from UNet
    :param label: 2d tensor: label / ground truth
    :return: iou
    '''''
    # Set a constant for IoU calculation
    c = 1e-8

    sig = nn.Sigmoid()
    output = sig(output)
    # Binarize the output and target
    output = (output > 0.5).float()
    # Flatten the output and target
    output = output.view(-1)
    label = label.view(-1)

    TP = torch.sum(output * label)
    FP = torch.sum(output * (1 - label))
    FN = torch.sum((1 - output) * label)

    # Calculate the Intersection over Union (IoU)
    iou = TP / (TP + FP + FN + c)  # Adding a small epsilon to avoid division by zero
    return iou


def categorical_dice(output, label):
    '''
    Calculate Dice accurancy
    :param output: 2d tensor: output from UNet
    :param label: 2d tensor: label / ground truth
    :return: dice value 
    '''''
    smooth = 1.

    sig = nn.Sigmoid()
    output = sig(output)

    output = (output > 0.5).float()

    m1 = output.view(-1)
    m2 = label.view(-1)

    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# PyTorch from https://www.kaggle.com/code/bigironsphere?scriptVersionId=68471013&cellId=4
class DiceLoss(nn.Module):
    '''
    Dice Loss for semantic segmentation tasks.
    '''''
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        # Return Dice loss
        return 1 - dice


#PyTorch https://www.kaggle.com/code/bigironsphere?scriptVersionId=68471013&cellId=7
class DiceBCELoss(nn.Module):
    '''
    Dice-BCE Loss for semantic segmentation tasks.
    '''''
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Calculate Dice-BCE
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE


def getName(val_input_dir):
    '''
    Retrieve all file names within the folder.
    '''''
    file_names = os.listdir(val_input_dir)
    # Save the file names to a list
    name_list = []
    for file_name in file_names:
        # Skip the .DS_Store files if any
        if '.DS_Store' in file_name:
            continue
        name_list.append(file_name)
    return name_list


def save_out_label(out, index, name_list, val_output_dir):
    '''
    Save the segmented mask for shadow of segmentation approach
    :param out: 4D tensor
    :param index: index of data for saving files
    :param name_list: list of names of file for saving files
    :param val_output_dir: directory for saving files
    :return: None
    '''''
    # use this when applying Sigmoid for segmentation
    threshold = 0.5
    # apply Sigmoid to ensure the range of output is [0, 1]
    m = nn.Sigmoid()
    out = m(out)
    out_thresholded = (out > threshold).float()

    # squeeze the output from a 4D tensor to 2D for visualization
    out_2d = out_thresholded.squeeze().squeeze()
    trans = transforms.ToPILImage()
    mask_img = trans(out_2d)
    # mask_img.imsave(val_output_dir + str(index) + ".tif", out_2d.numpy())
    mask_img = mask_img.convert('L')
    mask_img.save(val_output_dir + name_list[index])
    return

def plot_history(train_acc, train_loss, val_acc, val_loss, result_dir):
    '''
    Plot the figures of "model loss" and "model iou"
    :param acc: np array: model iou
    :param loss: np array: model loss
    :param result_dir: path to save the figures
    '''''
    # model accurancy
    plt.plot(train_acc, color='r', label="train_acc", marker='.')
    plt.plot(val_acc, color='orange', label="val_acc", marker='.')
    # set attributes
    plt.title('Average Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(title='Average Accuracy:', loc='center left')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    # model loss
    plt.plot(train_loss, color='blue', label="train_loss", marker='.')
    plt.plot(val_loss, color='green', label="val_loss", marker='.')
    # set attributes
    plt.title('Average Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(title='Average Loss:', loc='center left')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()