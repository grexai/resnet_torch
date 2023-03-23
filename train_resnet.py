import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

"""PARAMETERS"""

use_wandb = False
use_cuda = True
batch_size = 128
learning_rate = 5e-2
num_classes = 131
n_epochs = 5
dataset_path = "D:/datasets/archive/fruits-360_dataset/fruits-360/"

if use_wandb:
    project_name = f'resnet{50}'
    wandb.init(project=project_name)

transforms = transforms.Compose(
    [
        transforms.ToTensor()
        # transforms.Resize(64)
    ])

train_dataset = datasets.ImageFolder(root=f'{dataset_path}Training', transform=transforms)
test_dataset = datasets.ImageFolder(root=f'{dataset_path}Test', transform=transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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


#
# images, labels = next(iter(train_dataloader))
# print("images-size:", images.shape)
#
# out = torchvision.utils.make_grid(images)
# print("out-size:", out.shape)

# imshow(out, title=[train_dataset.classes[x] for x in labels])


net = models.resnet50(pretrained=True)
net = net.cuda() if device else net.cpu()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_classes)
net.fc = net.fc.cuda() if use_cuda else net.fc

print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
net.train()
print(device)
for epoch in range(1, n_epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
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
        if (batch_idx) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)

    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')

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
        wandb.log({"validation loss": np.mean(val_loss)})
        wandb.log({f"validation acc": (100 * correct_t / total_t)})
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet_best.pt')
    net.train()

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
