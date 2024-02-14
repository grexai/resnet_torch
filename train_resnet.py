from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
# from mpmath.identification import transforms
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR


"""PARAMETERS"""

use_wandb = False
use_cuda = True
batch_size = 512
learning_rate = 1e-3
num_classes = 5
n_epochs = 100
dataset_path = "D:/datasets/archive/fruits-360_dataset/fruits-360/"
dataset_path = "e:/DVP3/merged/"
dataset_path = "/storage01/grexai/datasets/DVP3/Mitotic5class_merged/"
if use_wandb:
    project_name = f'resnet{50}_for_mitotic'
    wandb.init(project=project_name)
else:
    experiment = Experiment(api_key="cfPvvHqzmKpiAK5gljgCjdJCQ", project_name="ResNet50_mitotic5class")
experiment.set_name("ResNet mitotic 5 Class aug")
transforms_train = transforms.Compose(
    [
        # transforms.Resize(64)
        transforms.RandomAffine(degrees=(30, 70), translate=(0.05, 0.15), scale=(0.95, 1.05)),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.9, 1.1)),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5),
        transforms.ColorJitter(brightness=.8, hue=.1),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])


transforms_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop(128),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])


train_dataset = datasets.ImageFolder(root=f'{dataset_path}train', transform=transforms_train)
test_dataset = datasets.ImageFolder(root=f'{dataset_path}val', transform=transforms_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if not use_cuda:
    device = 'cpu'


def imshow(inp, title=None):
    inp = inp.cpu() if device else inp
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()
#
# images, labels = next(iter(train_dataloader))
# print("images-size:", images.shape)
#
# out = torchvision.utils.make_grid(images)
# print("out-size:", out.shape)

# imshow(out, title=[train_dataset.classes[x] for x in labels])


# Load the pre-trained ResNet50 model
net = models.resnet50(pretrained=True)

# Move the model to CUDA if available
if use_cuda:
    net = net.cuda()
else:
    net = net.cpu()

# Modify the last fully connected layer
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_classes)

# Move the model to CUDA if available
if use_cuda:
    net.fc = net.fc.cuda()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
# optimizer = optim.SGD(net.parameters(), learning_rate, momentum=0.9)
optimizer = optim.Adam(net.parameters(), learning_rate)
#
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, num_classes)
# net.fc = nn.Softmax(num_classes)
# net.fc = net.fc.cuda() if use_cuda else net.fc
# if use_cuda:
#     net.fc =net.fc.cuda()
# else:
#     net.fc = net.fc
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
net.train()
print(device)


scheduler = MultiStepLR(optimizer, milestones=[20,50,80], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=n_epochs)
#print(scheduler.get_last_lr())
validation_acc = 0
train_accuracy = 0
train_loss_current = []
for epoch in tqdm(range(1, n_epochs + 1), desc=f"learning rate {optimizer.param_groups[-1]['lr']:.6f}"):
    running_loss = 0.0
    correct = 0
    total = 0

    dataset_len = len(train_dataloader)
    for batch_idx, (data_, target_) in tqdm(enumerate(train_dataloader),maxinterval=dataset_len):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()

        outputs = net(data_)
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()
        
        
        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)

        correct += torch.sum(pred == target_).item()
        total += target_.size(0)
        if use_wandb:
            wandb.log({"train loss": loss.item()})
        # if (batch_idx) % 20 == 0:
        #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #           .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    train_accuracy = 100 * correct / total
    train_loss_current = running_loss / total_step
    # print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')

    batch_loss = 0
    total_t = 0
    correct_t = 0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in (test_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _, pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t == target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss / len(test_dataloader))
        network_learned = batch_loss < valid_loss_min
        if use_wandb:
            wandb.log({"validation loss": np.mean(val_loss)})
            wandb.log({f"validation acc": (100 * correct_t / total_t)})
        else:
            experiment.log_metric("train loss", loss.item())
            experiment.log_metric("validation loss", np.mean(val_loss))
            experiment.log_metric("validation acc", (100 * correct_t / total_t))
        validation_acc = 100 * correct_t / total_t
        
        # print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet50_aug_best.pt')
    net.train()
    scheduler.step()



# fig = plt.figure(figsize=(20,10))
# plt.title("Train-Validation Accuracy")
# plt.plot(train_acc, label='train')
# plt.plot(val_acc, label='validation')
# plt.xlabel('num_epochs', fontsize=12)
# plt.ylabel('accuracy', fontsize=12)
# plt.legend(loc='best')

#
# def visualize_model(net, num_images=4):
#     images_so_far = 0
#     fig = plt.figure(figsize=(15, 10))
#
#     for i, data in enumerate(test_dataloader):
#         inputs, labels = data
#         if use_cuda:
#             inputs, labels = inputs.cuda(), labels.cuda()
#         outputs = net(inputs)
#         _, preds = torch.max(outputs.data, 1)
#         preds = preds.cpu().numpy() if use_cuda else preds.numpy()
#         for j in range(inputs.size()[0]):
#             images_so_far += 1
#             ax = plt.subplot(2, num_images // 2, images_so_far)
#             ax.axis('off')
#             ax.set_title('predictes: {}'.format(test_dataset.classes[preds[j]]))
#             imshow(inputs[j])
#
#             if images_so_far == num_images:
#                 return
#
#
# plt.ion()
# visualize_model(net)
# plt.ioff()
