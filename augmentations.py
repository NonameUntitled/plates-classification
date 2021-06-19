import shutil

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
from PIL import Image
from collections import defaultdict, Counter

TRAIN_DIR = r'data/train/processed_train'
VAL_DIR = r'data/train/processed_val'
TEST_DIR = r'data/processed_test'

BATCH_SIZE = 16  # Количество изображений в батче

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PlateClassification(nn.Module):
    def __init__(self):
        super(PlateClassification, self).__init__()
        self.model = models.resnet152(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.to(DEVICE)

    def forward(self, x):
        return self.model(x)


def train_model(model, loss, optimizer, scheduler, num_epochs):
    loss_hist = {'train': [], 'val': []}
    acc_hist = {'train': [], 'val': []}

    for i in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader

                model.train()
            else:
                dataloader = val_dataloader
                model.eval()

            running_loss = 0.
            running_acc = 0.

            for X_batch, y_batch in tqdm(dataloader):
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X_batch)
                    loss_value = loss(y_pred, y_batch)
                    y_pred_class = y_pred.argmax(dim=1)

                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss_value.item()
                running_acc += (y_pred_class == y_batch.data).float().mean().data.cpu().numpy()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ', end='')

            torch.save(model.state_dict(), f'./weights/model_{i:20d}.torch')

            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)

    return model, loss_hist, acc_hist


if __name__ == '__main__':
    # filenames = os.listdir(os.path.join(TRAIN_DIR, 'cleaned'))
    # np.random.shuffle(filenames)
    # for idx, filename in enumerate(filenames):
    #     if idx % 6 == 5:
    #         old_filepath = os.path.join(TRAIN_DIR, 'cleaned', filename)
    #         new_path = os.path.join(VAL_DIR, 'cleaned', filename)
    #         shutil.move(old_filepath, new_path)
    # filenames = os.listdir(os.path.join(TRAIN_DIR, 'dirty'))
    # np.random.shuffle(filenames)
    # for idx, filename in enumerate(filenames):
    #     if idx % 6 == 5:
    #         old_filepath = os.path.join(TRAIN_DIR, 'dirty', filename)
    #         new_path = os.path.join(VAL_DIR, 'dirty', filename)
    #         shutil.move(old_filepath, new_path)
    # exit(0)

    model = PlateClassification()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    train_transforms = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.09, p=0.75, interpolation=3, fill=255),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(hue=(-0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.1, p=0.8, interpolation=3, fill=255),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(hue=(-0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR, val_transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=BATCH_SIZE
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=BATCH_SIZE
    )

    model, loss_history, accuracy_history = train_model(model, loss_function, optimizer, scheduler, 40)

    # Test
    img_info = defaultdict(list)
    for filename in tqdm(os.listdir(TEST_DIR)):
        img_id = filename[filename.find('_') + 1: filename.find('_') + 5]
        filepath = os.path.join(TEST_DIR, filename)
        img = Image.open(filepath)
        img = test_transforms(img)
        img = torch.reshape(img, (1, *img.shape)).to(DEVICE)
        res = model(img).cpu().argmax(dim=1).item()
        img_info[img_id].append(res)

    mapping = {}
    for key, lst in sorted(list(img_info.items()), key=lambda x: x[0]):
        counter = Counter(lst)
        if counter[0] >= counter[1]:
            mapping[key] = 'cleaned'
        else:
            mapping[key] = 'dirty'

    data = []
    for key, item in mapping.items():
        data.append([int(key), item])

    import pandas as pd

    df = pd.DataFrame(data, columns=['id', 'label'])
    df.to_csv('./results.csv', index=False)

    plt.rcParams['figure.figsize'] = (14, 7)
    for experiment_id in accuracy_history.keys():
        plt.plot(accuracy_history[experiment_id], label=experiment_id)
    plt.legend(loc='upper left')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch num', fontsize=15)
    plt.ylabel('Accuracy value', fontsize=15)
    plt.grid(linestyle='--', linewidth=0.5, color='.7')
    plt.show()

    plt.rcParams['figure.figsize'] = (14, 7)
    for experiment_id in loss_history.keys():
        plt.plot(loss_history[experiment_id], label=experiment_id)
    plt.legend(loc='upper left')
    plt.title('Model Loss')
    plt.xlabel('Epoch num', fontsize=15)
    plt.ylabel('Loss function value', fontsize=15)
    plt.grid(linestyle='--', linewidth=0.5, color='.7')
    plt.show()
