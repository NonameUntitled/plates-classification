from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from constants import *
from model import PlateClassifier


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

                running_loss += loss_value.item()
                running_acc += (y_pred_class == y_batch.data).float().mean().data.cpu().numpy()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            if phase == 'train':
                scheduler.step()
            if phase == 'val':
                torch.save(model.state_dict(), f'./weights/model_{i}.torch')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ', end='')

            loss_hist[phase].append(epoch_loss)
            acc_hist[phase].append(epoch_acc)

    return model, loss_hist, acc_hist


if __name__ == '__main__':

    model = PlateClassifier()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)

    image_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3, fill=255),
                transforms.RandomChoice([
                    transforms.CenterCrop(180),
                    transforms.CenterCrop(160),
                    transforms.CenterCrop(140),
                    transforms.CenterCrop(120),
                    transforms.Compose([
                        transforms.CenterCrop(280),
                        transforms.Grayscale(3),
                    ]),
                    transforms.Compose([
                        transforms.CenterCrop(200),
                        transforms.Grayscale(3),
                    ]),
                ]),
                transforms.Resize((224, 224)),
                transforms.ColorJitter(hue=(0.1, 0.2)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'val':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3, fill=255),
                transforms.RandomChoice([
                    transforms.CenterCrop(180),
                    transforms.CenterCrop(160),
                    transforms.CenterCrop(140),
                    transforms.CenterCrop(120),
                    transforms.Compose([
                        transforms.CenterCrop(280),
                        transforms.Grayscale(3),
                    ]),
                    transforms.Compose([
                        transforms.CenterCrop(200),
                        transforms.Grayscale(3),
                    ]),
                ]),
                transforms.Resize((224, 224)),
                transforms.ColorJitter(hue=(0.1, 0.2)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_dataset = torchvision.datasets.ImageFolder(PROCESSED_TRAIN_PATH, image_transforms['train'])
    val_dataset = torchvision.datasets.ImageFolder(PROCESSED_VAL_PATH, image_transforms['val'])

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=BATCH_SIZE
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=BATCH_SIZE
    )

    model, loss_history, accuracy_history = train_model(model, loss_function, optimizer, scheduler, 40)

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

    # Test
    # model.load_state_dict(torch.load('weights/model_39.torch'))
    img_info = defaultdict(list)
    for filename in tqdm(os.listdir(PROCESSED_TEST_PATH)):
        img_id = filename[filename.find('_') + 1: filename.find('_') + 5]
        filepath = os.path.join(PROCESSED_TEST_PATH, filename)
        img = Image.open(filepath)
        img = image_transforms['test'](img)
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
