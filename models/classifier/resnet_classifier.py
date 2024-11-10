from torchvision import models
import gc
import time

import numpy as np
import pandas as pd
import cv2
import timm
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def get_crop(image: np.array, x: float, y: float, w: float, h: float):
    hg, wg, _ = image.shape
    x, y, w, h = int(x * wg), int(y * hg), int(w * wg), int(h * hg)
    img = cv2.resize(image[y-h//2:y+h//2, x-w//2:x+w//2], (512, 512), interpolation = cv2.INTER_AREA)
    return img

def all_crops(images_path: str, annot_path: str):
    annot = pd.read_csv(annot_path)
    crops = []
    target = []
    for idx, row in annot.iterrows():
        temp_img = cv2.imread(images_path + row["Name"])
        # print(temp_img, *map(float, row["Bbox"].split(",")), images_path + row["Name"])
        crop = get_crop(temp_img, *map(float, row["Bbox"].split(",")))
        crops.append(crop)
        target.append(row["Class"])
    return np.array(crops), np.array(target)

# all_crops("data/images/", "data/annotation.csv")#
crops, target = all_crops("data/images/", "data/annotation.csv")# all_crops("D:\\hacks\\train_data_minprirodi\\images\\", "D:\\hacks\\train_data_minprirodi\\annotation.csv")
print(len(crops))
SEED = 42
NUM_CLASSES = 2 # df.nunique()['label']
NUM_WORKERS = 2
BATCH_SIZE = 8

LR = 0.0001


def transform_train():
    transform = [
        A.Resize(512,512,p=1),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.CoarseDropout(p=0.5),
        ToTensorV2(p=1.0)
    ]
    return A.Compose(transform)


# Validation (and test) images should only be resized.
def transform_valid():
    transform = [
        A.Resize(512,512,p=1),
        ToTensorV2(p=1.0)
    ]
    return A.Compose(transform)

class ResNet101Classifier(nn.Module):
    def __init__(self, n_out):
        super(ResNet101Classifier, self).__init__()
        # Define model
        self.resnet = models.resnet101(pretrained=True)

    def forward(self, x):
        return self.resnet(x)


class AnimalsDataset(Dataset):
    def __init__(self, crops, target, transforms=None, give_label=True):
        """Performed only once when the Dataset object is instantiated.
        give_label should be False for test data
        """
        super().__init__()
        self.images = crops
        self.targets = target
        self.transforms = transforms

    def __len__(self):
        """Function to return the number of records in the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """Function to return samples corresponding to a given index from a dataset
        """
        # Load images
        img = self.images[index]
        # img /= 255.0 # Normalization

        # Transform images
        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, self.targets[index]



def create_dataloader(images, targets, trn_idx, val_idx):

    train_images = images[trn_idx]
    train_targets = targets[trn_idx]
    val_images = images[val_idx]
    val_targets = targets[val_idx]

    # Dataset
    train_datasets = AnimalsDataset(train_images, train_targets, transforms=transform_train())
    valid_datasets = AnimalsDataset(val_images, val_targets, transforms=transform_valid())

    # Data Loader
    train_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    return train_loader, valid_loader

FOLD_NUM = 3 # For cross-validation
EPOCHS = 10
folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED) \
    .split(np.arange(len(target)), np.array(target))

# For Visualization

print("wh")

if __name__ == "__main__":
    train_acc_list = []
    valid_acc_list = []
    train_loss_list = []
    valid_loss_list = []

    print(folds, len(crops))

    device = "cpu" # "cuda"
    for fold, (trn_idx, val_idx) in enumerate(folds):
        print(f'==========Cross Validation Fold {fold + 1}==========')
        # Load Data
        train_loader, valid_loader = create_dataloader(crops, target, trn_idx, val_idx)

        # Load model, loss function, and optimizing algorithm
        model = ResNet101Classifier(NUM_CLASSES).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # For Visualization
        train_accs = []
        valid_accs = []
        train_losses = []
        valid_losses = []

        # Start training
        best_acc = 0
        for epoch in range(EPOCHS):
            time_start = time.time()
            print(f'==========Epoch {epoch + 1} Start Training==========')
            model.train()

            epoch_loss = 0
            epoch_accuracy = 0

            print(train_loader)

            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for step, (img, label) in pbar:
                img = img.to(device).float()
                label = label.to(device).long()

                output = model(img)
                print("output: ", output)
                print()
                print("label: ", label)
                loss = loss_fn(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)

            print(f'==========Epoch {epoch + 1} Start Validation==========')
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                val_labels = []
                val_preds = []

                pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
                for step, (img, label) in pbar:
                    img = img.to(device).float()
                    label = label.to(device).long()

                    val_output = model(img)
                    val_loss = loss_fn(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)

                    val_labels += [label.detach().cpu().numpy()]
                    val_preds += [torch.argmax(val_output, 1).detach().cpu().numpy()]

                val_labels = np.concatenate(val_labels)
                val_preds = np.concatenate(val_preds)

            # print result from this epoch
            exec_t = int((time.time() - time_start) / 60)
            print(
                f'Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} / Exec time {exec_t} min\n'
            )

            # For visualization
            train_accs.append(epoch_accuracy.cpu().numpy())
            valid_accs.append(epoch_val_accuracy.cpu().numpy())
            train_losses.append(epoch_loss.detach().cpu().numpy())
            valid_losses.append(epoch_val_loss.detach().cpu().numpy())

        train_acc_list.append(train_accs)
        valid_acc_list.append(valid_accs)
        train_loss_list.append(train_losses)
        valid_loss_list.append(valid_losses)
        torch.save(model.state_dict(), f"resnet_weights_class{fold}.pth")
        del model, optimizer, train_loader, valid_loader, train_accs, valid_accs, train_losses, valid_losses
        gc.collect()
        torch.cuda.empty_cache()

    print(train_acc_list, train_loss_list, valid_acc_list, valid_loss_list)