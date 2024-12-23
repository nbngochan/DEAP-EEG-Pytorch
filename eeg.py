import torch
from torch.utils.data import DataLoader, random_split
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
from braindecode.models import EEGNetv1
from braindecode.util import set_random_seeds
import lightning as L
from torchmetrics.functional import accuracy
import os
import time
import logging
import torch.nn as nn

os.makedirs('./examples_vanilla_torch/log', exist_ok=True)
logger = logging.getLogger('Training models with vanilla PyTorch')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./examples_vanilla_torch/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---- DATASET CONFIGURATION ---- #
root_path = '/mnt/inaisfs/user-fs/imsp/Han/dataset/DEAP/data_preprocessed_python/'
dataset = DEAPDataset(root_path=root_path,
                      io_path='./phase1/deap',
                      online_transform=transforms.Compose([
                          transforms.To2d(),
                          transforms.ToTensor(),
                      ]),
                      label_transform=transforms.Compose([
                          transforms.Select(['valence', 'arousal']),
                          transforms.Binary(5.0),
                          transforms.BinariesToCategory()
                      ]))

from torcheeg.model_selection import KFoldGroupbyTrial

k_fold = KFoldGroupbyTrial(n_splits=5,
                           split_path='./phase1/split',
                           shuffle=True,
                           random_state=44)

from torch.utils.data import DataLoader
from torcheeg.models import EEGNet


import pytorch_lightning as pl


device = "cuda" if torch.cuda.is_available() else "cpu"

# training process
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    record_step = int(len(dataloader) / 10)

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % record_step == 0:
            loss, current = loss.item(), batch_idx * len(X)
            logger.info(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


# validation process
def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    logger.info(
        f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n"
    )

    return correct, loss

from torcheeg.model_selection import train_test_split
loss_fn = nn.CrossEntropyLoss()
batch_size = 64

test_accs = []
test_losses = []

for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
    # initialize model
    model = EEGNet(chunk_size=128,
               num_electrodes=32,
               dropout=0.5,
               kernel_1=64,
               kernel_2=16,
               F1=8,
               F2=16,
               D=2,
               num_classes=4).to('cuda')
    
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4)  # official: weight_decay=5e-1
    # split train and val
    train_dataset, val_dataset = train_test_split(
        train_dataset,
        test_size=0.2,
        split_path=f'./phase1/split{i}',
        shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    epochs = 50
    best_val_acc = 0.0
    for t in range(epochs):
        train_loss = train(train_loader, model, loss_fn, optimizer)
        val_acc, val_loss = valid(val_loader, model, loss_fn)
        # save the best model based on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f'./phase1/model{i}.pt')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load the best model to test on test set
    model.load_state_dict(torch.load(f'./phase1/model{i}.pt'))
    test_acc, test_loss = valid(test_loader, model, loss_fn)

    # log the test result
    logger.info(
        f"Test Error {i}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}"
    )

    test_accs.append(test_acc)
    test_losses.append(test_loss)

# log the average test result on cross-validation datasets
logger.info(
    f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}"
)
