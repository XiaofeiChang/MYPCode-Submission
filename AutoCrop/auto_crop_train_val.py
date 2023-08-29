'''
This file contain training and validation of UNet.
'''''
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import sys
from AutoCrop import auto_crop_model_utility
np.set_printoptions(threshold=sys.maxsize)  # used to print full numpy array
from auto_crop_Dataset import TrainValDataset
from auto_crop_model_UNet import UNet
from auto_crop_model_utility import *
from sklearn.model_selection import train_test_split
from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
import requests
url = 'https://github.com/qubvel/segmentation_models.pytorch'
response = requests.get(url, verify=False)
import os
os.environ['REQUESTS_CA_BUNDLE'] = 'https://github.com/qubvel/segmentation_models.pytorch'



epoch = 64
batch_size = 5
lr = 0.001

# train
train_size = 113 // batch_size + 1
lossList_temp = []
accList_temp = []
lossList = []
accList = []

# validation
val_size = 27
val_lossList = []
val_accList = []
val_acc_last_epoch_list = []
record_val_acc_List = []

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)
# Load data
train_input_dir, train_label_dir, val_input_dir, val_label_dir, val_output_dir = getPath()
train_dataset = TrainValDataset(train_input_dir, train_label_dir)
val_dataset = TrainValDataset(val_input_dir, val_label_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Loss
# loss_func = nn.CrossEntropyLoss()
# loss_func = DiceLoss()
loss_func = DiceBCELoss()

# SGD with Momentum
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # SGD
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6, 15, 22, 30, 36, 41, 47],
                                      gamma=0.9)  # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
# Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

global_step = 0

# Begin training
for i in range(epoch):
    cnt = 0
    j = 0

    for image, label in train_loader:
        image = image.to(device)
        image.requires_grad = True
        label = label.to(device)

        output = model(image)

        loss = loss_func(output, label)
        dice = IoU(output, label)

        optimizer.zero_grad()  # zero the parameter gradients, do this before loss.backward()
        loss.backward()  # Backpropagation
        optimizer.step()  # update parameters according to GD
        cnt += 1
        global_step += 1

        # save the data for ploting
        lossList_temp.append(loss.item())
        accList_temp.append(dice.cpu().numpy())

        if cnt == train_size:
            acc_avg = np.mean(np.asarray(accList_temp))
            loss_avg = np.mean(np.asarray(lossList_temp))
            accList.append(acc_avg)
            lossList.append(loss_avg)
            lossList_temp = []
            accList_temp = []

            # Initialize the loss and acc for val
            val_loss_temp = []
            val_acc_temp = []
            val_cnt = 0

            # do validation
            with torch.no_grad():
                for val_image, val_label in val_loader:
                    val_image = val_image.to(device)
                    val_label = val_label.to(device)

                    # predict output for validation
                    val_output = model(val_image)

                    # accumulate loss
                    val_loss_single = loss_func(val_output, val_label).item()
                    val_loss_temp.append(val_loss_single)

                    # accumulate acc
                    val_loss_single = IoU(val_output, val_label).cpu().numpy()
                    val_acc_temp.append(val_loss_single)

                    # for the last epoch, use a list to record the max_acc, avg_acc, mean_acc, std_acc
                    if i == epoch-1:
                        val_acc_last_epoch_list.append(val_acc_temp)

                    val_cnt += 1

                    print("Validation", f'epoch: {i}, cnt: {val_cnt}, loss: {val_loss_single}, dice: {val_loss_single}')

                    if val_cnt == val_size:
                        val_loss_avg = np.mean(np.asarray(val_loss_temp))
                        val_acc_avg = np.mean(np.asarray(val_acc_temp))
                        val_lossList.append(val_loss_avg)
                        val_accList.append(val_acc_avg)
                        val_cnt = 0

                        # record the averaege of acc of the last epoch
                        if i == epoch-1:
                            val_acc_avg_record = val_acc_avg

        # used for Adam
        # adam_lr = optimizer.param_groups[0]['lr']
        # print("Train", f'epoch: {i}, cnt: {cnt}, loss: {loss.item()}, dice: {dice}, lr: {adam_lr}')

        # used for SGD
        print("Train", f'epoch: {i}, cnt: {cnt}, loss: {loss.item()}, dice: {dice}, lr: {scheduler.get_last_lr()}')

    scheduler.step()

# record of training
val_acc_max_record = np.max(val_acc_last_epoch_list)
val_acc_mean_record = np.mean(val_acc_last_epoch_list)
val_acc_std_record = np.std(val_acc_last_epoch_list)

# max, average, mean, std
record_val_acc_List = [val_acc_max_record, val_acc_avg_record, val_acc_mean_record, val_acc_std_record]

np.savetxt("max,average,mean,std.txt", record_val_acc_List)
# The training acc and loss of each epoch
np.savetxt("acc_unet.txt", accList)
np.savetxt("loss_unet.txt", lossList)
# The validation acc and loss of each epoch
np.savetxt("val_acc_unet.txt", val_accList)
np.savetxt("val_loss_unet.txt", val_lossList)
plot_history(accList, lossList, val_accList, val_lossList, "./")

# save the model
torch.save(model.state_dict(), './auto_crop_unet.pth')
