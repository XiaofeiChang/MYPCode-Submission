'''
This file is includes the data loader for UNet
'''''
import os
import matplotlib
import torchvision as torchvision
from torch.utils.data import Dataset
from PIL import Image
matplotlib.use('Agg')



class TestMain(Dataset):
    '''
    Data Loader for Testing (auto_crop_test.py)
    '''''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    def __init__(self, data_dir):
        '''
        Load data for testing that integrated to main pipeline of this project
        :param data_dir: path of input directory
        '''''
        self.data_dir = data_dir
        self.datas = self.getData(data_dir)

    def __getitem__(self, index):
        imgPath = self.datas[index]
        img = Image.open(self.data_dir + imgPath)
        img_Tensor = self.transform(img).float()
        return img_Tensor

    def __len__(self):
        return len(self.datas)

    def getData(self, data_dir):
        '''
        Load the input images
        :param data_dir: path of input directory
        :return: path of input files
        '''''
        files = []
        for file in os.listdir(data_dir):
            # Skip the .DS_Store files if any
            if '.DS_Store' in file:
                continue
            files.append(file)
        return files


class TrainValDataset(Dataset):
    '''
    Data Loader for Training and Validation (auto_crop_train_val.py)
    '''''
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    def __init__(self, data_dir, label_dir):
        '''
        Load data for UNet model training and validation and testing
        :param data_dir: path of input directory
        :param label_dir: path of label/ground truth/target directory
        '''''
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.datas = self.getData(data_dir)
        self.labels = self.getLabel(data_dir)

    def __getitem__(self, index):
        imgPath = self.datas[index]
        labelPath = self.labels[index]

        img = Image.open(self.data_dir + imgPath)
        label = Image.open(self.label_dir + labelPath)

        img_Tensor = self.transform(img).float()
        label_Tensor = self.transform(label.convert('L')).float()
        # binarize the tensor to ensure binary classes
        label_Tensor = (label_Tensor > 0.1).float()
        return img_Tensor, label_Tensor

    def __len__(self):
        return len(self.datas)

    def getData(self, data_dir):
        '''
        Load the input images
        :param data_dir: path of input directory
        :return: path of input files
        '''''
        files = []
        for file in os.listdir(data_dir):
            # Skip the .DS_Store files if any
            if '.DS_Store' in file:
                continue
            files.append(file)
        return files

    def getLabel(self, data_dir):
        '''
        Load the labels
        :param data_dir: path of label directory
        :return: path of label files
        '''''
        labels = []
        for label in os.listdir(data_dir):
            # Skip the .DS_Store files if any
            if '.DS_Store' in label:
                continue
            labels.append(label)
        return labels
