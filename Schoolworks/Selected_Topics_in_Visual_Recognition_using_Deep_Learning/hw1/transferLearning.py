from __future__ import print_function, division

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

from PIL import Image
from sklearn.model_selection import train_test_split

plt.ion()   # interactive mode


torch.cuda.empty_cache()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

"""
   split training dataset into train and val
"""
"""
from shutil import copy

os.mkdir('/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/train')
os.mkdir('/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/val')

path = '/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/training_labels.txt'

imageID = []
classID = []

with open(path) as f:
    for line in f.readlines():
        s = line.split(' ')
        imageID.append(s[0])
        classID.append(s[1][:-1])

f.close()

seed = 3557
ratio = 0.75

train_imageID, val_imageID = train_test_split(imageID, random_state = seed, train_size = ratio)
train_classID, val_classID = train_test_split(classID, random_state = seed, train_size = ratio)

for i in range(len(train_classID)):
    folder = '/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/train/' + train_classID[i]
    if not os.path.exists(folder):
        os.mkdir(folder)
    img = '/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/training_images/' + train_imageID[i]
    copy(img, folder)


for i in range(len(val_classID)):
    folder = '/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/val/' + val_classID[i]
    if not os.path.exists(folder):
        os.mkdir(folder)
    img = '/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/training_images/' + val_imageID[i]
    copy(img, folder)
"""



"""
   
"""
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

################################################################################
#                       data augmentation and dataloaders                      0
################################################################################
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(416),    # 224
        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),
        #transforms.RandomAdjustSharpness(2),
        #transforms.RandomAutocontrast(),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.ColorJitter(),

        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),    # 256
        transforms.CenterCrop(416),    # 224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}


data_dir = '/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2,    # 4
                                             shuffle=True, num_workers=4)    # 4
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
#print(class_names)


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




"""
   
"""
#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.

################################################################################
#                                 select model                                 1
################################################################################
#model = models.resnet18(pretrained=True)
#model = models.resnet50(pretrained=True)
#model = models.resnet101(pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=True)

num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()


################################################################################
#                         setup learning rate schedule                         2
################################################################################
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
#step_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0, last_epoch=-1)
#step_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=40)


################################################################################
#                                 setup epochs                                 3
################################################################################
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=400)




"""
   save model
"""
################################################################################
#                              rename output file                              4
################################################################################
FILE = '/home/yuhsi44165/NYCU/G2/VRDL/HW1/results/resnest269_d3e2.pth'
torch.save(model.state_dict(), FILE)




"""
   load model
"""
################################################################################
#                                 select model                                 5
################################################################################
#model = models.resnet18(pretrained=True)
#model = models.resnet50(pretrained=True)
#model = models.resnet101(pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(FILE))
model.eval()




"""
   prediction
"""
# Create a preprocessing pipeline
preprocess = transforms.Compose([
        #transforms.Resize(256),
        transforms.CenterCrop(416),    # 224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

with open('/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/testing_img_order.txt') as f:
     test_images = [x.strip() for x in f.readlines()]  # all the testing images

submission = []
for img_name in test_images:  # image order is important to your result

    img = Image.open('/home/yuhsi44165/NYCU/G2/VRDL/HW1/2021VRDL_HW1_datasets/testing_images/' + img_name)
    img_preprocessed = preprocess(img)
    batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
    out = model(batch_img_tensor)

    _, index = torch.max(out, 1)

    predicted_class = class_names[index[0]]  # the predicted category
    #print(predicted_class)

    submission.append([img_name, predicted_class])

np.savetxt('answer.txt', submission, fmt='%s')


