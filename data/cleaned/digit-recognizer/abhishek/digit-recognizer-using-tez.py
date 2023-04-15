

import os
import albumentations
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn import metrics, model_selection
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.utils import seed_everything

class args:
    input = 'data/input/digit-recognizer/'
    model_name = 'resnet50'
    learning_rate = 0.01
    batch_size = 64
    epochs = 25
    output = '.'
    accumulation_steps = 1

class DigitRecognizerDataset:

    def __init__(self, df, augmentations):
        self.df = df
        self.targets = df.label.values
        self.df = self.df.drop(columns=['label'])
        self.augmentations = augmentations
        self.images = self.df.to_numpy(dtype=np.float32).reshape((-1, 28, 28))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        targets = self.targets[item]
        image = self.images[item]
        image = np.expand_dims(image, axis=0)
        return {'image': torch.tensor(image, dtype=torch.float), 'targets': torch.tensor(targets, dtype=torch.long)}

class DigitRecognizerModel(nn.Module):

    def __init__(self, model_name, num_classes, learning_rate, n_train_steps):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_train_steps = n_train_steps
        self.model = timm.create_model(model_name, pretrained=True, in_chans=1, num_classes=num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        targets = targets.cpu().detach().numpy()
        acc = metrics.accuracy_score(targets, outputs)
        acc = torch.tensor(acc, device=device)
        return {'accuracy': acc}

    def optimizer_scheduler(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, verbose=True, mode='max', threshold=0.0001)
        return (opt, sch)

    def forward(self, image, targets=None):
        x = self.model(image)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(x, targets)
            metrics = self.monitor_metrics(x, targets)
            return (x, loss, metrics)
        return (x, 0, {})
seed_everything(42)
df = pd.read_csv(os.path.join(args.input, 'train.csv'))
test_df = pd.read_csv(os.path.join(args.input, 'test.csv'))
test_df.loc[:, 'label'] = 0
train_aug = albumentations.Compose([albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0)], p=1.0)
valid_aug = albumentations.Compose([albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0)], p=1.0)
(train_df, valid_df) = model_selection.train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'].values)
train_dataset = DigitRecognizerDataset(df=train_df, augmentations=train_aug)
valid_dataset = DigitRecognizerDataset(df=valid_df, augmentations=valid_aug)
test_dataset = DigitRecognizerDataset(df=test_df, augmentations=valid_aug)
n_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
model = DigitRecognizerModel(model_name=args.model_name, num_classes=df.label.nunique(), learning_rate=args.learning_rate, n_train_steps=n_train_steps)
model = Tez(model)
config = TezConfig(training_batch_size=args.batch_size, validation_batch_size=2 * args.batch_size, test_batch_size=2 * args.batch_size, gradient_accumulation_steps=args.accumulation_steps, epochs=args.epochs, step_scheduler_after='epoch', step_scheduler_metric='valid_accuracy', fp16=True)
es = EarlyStopping(monitor='valid_accuracy', model_path=os.path.join(args.output, 'model.bin'), patience=10, mode='max', save_weights_only=True)