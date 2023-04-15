import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[0:10000, :]
df = df.reset_index()
df.head()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, random_state=7, shuffle=True)
folds = df.copy()
for (f, (tr_idx, val_idx)) in enumerate(skf.split(folds, folds.label)):
    folds.loc[val_idx, 'fold'] = int(f)
folds['fold'] = folds['fold'].astype(int)
folds.groupby('fold').label.value_counts()
oof = df.copy()
class_cols = [str(x) for x in np.sort(df.label.unique())]
oof[class_cols] = 0
oof.set_index('index', inplace=True)
oof.head()
train_aug = None
val_aug = None

class Csn(Dataset):

    def __init__(self, train_df, augs=None):
        self.df = train_df
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_ids = self.df.iloc[idx]['index']
        image = self.df.iloc[idx, 2:].values.reshape((-1, 28, 28))
        if self.augs == True:
            transformed = self.augs(image=image)
            image = transformed['image']
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float)
        label = self.df.iloc[idx].label
        label = torch.tensor(label, dtype=torch.long)
        return (image, label, image_ids)
eg_data = Csn(df)
plt.imshow(eg_data[5][0].squeeze(0))

class Cnn(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, (3, 3))
        self.pool = nn.AdaptiveAvgPool2d((10, 10))
        self.fc1 = nn.Linear(16 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model1 = Cnn()
import os
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(train_loader, model, optimizer, criterion, e, epochs):
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()
    global_step = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for (step, (image, labels, _)) in loop:
        image = image.to(device)
        labels = labels.to(device)
        output = model(image)
        batch_size = labels.size(0)
        loss = criterion(output, labels)
        out = output.softmax(1)
        outputs = torch.argmax(out, dim=1).cpu().detach().numpy()
        targets = labels.cpu().detach().numpy()
        accuracy = accuracy_score(targets, outputs)
        losses.update(loss.item(), batch_size)
        scores.update(accuracy.item(), batch_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        loop.set_description(f'Epoch {e + 1}/{epochs}')
        loop.set_postfix(loss=loss.item(), accuracy=accuracy.item(), stage='train')
    return (losses.avg, scores.avg)

def val_one_epoch(loader, model, optimizer, criterion):
    losses = AverageMeter()
    scores = AverageMeter()
    model.eval()
    global_step = 0
    loop = tqdm(enumerate(loader), total=len(loader))
    for (step, (image, labels, _)) in loop:
        image = image.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            output = model(image)
        loss = criterion(output, labels)
        output = output.softmax(1)
        outputs = torch.argmax(output, dim=1).cpu().detach().numpy()
        targets = labels.cpu().detach().numpy()
        accuracy = accuracy_score(targets, outputs)
        losses.update(loss.item(), batch_size)
        scores.update(accuracy.item(), batch_size)
        loop.set_postfix(loss=loss.item(), accuracy=accuracy.item(), stage='valid')
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    return (losses.avg, scores.avg)

def fit(fold_n, training_batch_size=64, validation_batch_size=128):
    train_data = folds[folds.fold != fold_n].iloc[:, :-1]
    val_data = folds[folds.fold == fold_n].iloc[:, :-1]
    train_data = Csn(train_data)
    val_data = Csn(val_data)
    train_loader = DataLoader(train_data, shuffle=True, num_workers=0, batch_size=training_batch_size)
    valid_loader = DataLoader(val_data, shuffle=False, num_workers=0, batch_size=validation_batch_size)
    model = Cnn()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    epochs = 3
    best_acc = 0
    loop = range(epochs)
    for e in loop:
        (train_loss, train_accuracy) = train_one_epoch(train_loader, model, optimizer, criterion, e, epochs)
        print(f'For epoch {e + 1}/{epochs}')
        print(f'average train_loss {train_loss}')
        print(f'average train_accuracy {train_accuracy}')
        (val_loss, val_accuracy) = val_one_epoch(valid_loader, model, optimizer, criterion)
        scheduler.step(val_loss)
        print(f'avarage val_loss {val_loss}')
        print(f'avarage val_accuracy {val_accuracy}')
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            print(f'saving model for {best_acc}')
            torch.save(model.state_dict(), OUTPUT_DIR + f'Fold {fold_n} model with val_acc {best_acc}.pth')
    best_model = Cnn().to(device)
    best_model.load_state_dict(torch.load(OUTPUT_DIR + f'Fold {fold_n} model with val_acc {best_acc}.pth'))
    best_model.eval()
    for (inputs, _, imgids) in valid_loader:
        with torch.no_grad():
            inputs = inputs.to(device)
            output = best_model(inputs)
            softed = F.softmax(output, dim=1)
            oof.loc[imgids, class_cols] = softed.cpu().numpy()
for i in range(3):
    print(f'######### for Fold {i} ###########')
    fit(i)
o = oof[class_cols]
o['label'] = oof['label']
oof = o

oof.head()
oof_accuracy = accuracy_score(oof.label.values, np.argmax(oof[class_cols].values, axis=1))
oof_accuracy