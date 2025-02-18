import os
import time
import logging
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.models import EEGNet
from braindecode.models import EEGConformer
from models.bgru import BGRU
from torcheeg.model_selection import KFoldGroupbyTrial, train_test_split
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        X, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(dataloader, model, loss_fn):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(dataloader.dataset), total_loss / len(dataloader)


def main(args):
    if args.DWT:
        online_transform = transforms.Compose(
            [
                transforms.BaselineRemoval(),
                transforms.DWTDecomposition(),
                transforms.Lambda(lambda x: x.mean(axis=0)),
                transforms.MeanStdNormalize(axis=0),
                transforms.ToTensor(),
            ]
        )
        chunk_size = 64
    else:
        online_transform = transforms.Compose(
            [
                transforms.BaselineRemoval(),
                transforms.MeanStdNormalize(axis=0),
                transforms.ToTensor(),
            ]
        )
        chunk_size = 128
    # label
    if args.valence:
        label_transform = transforms.Compose(
            [
                transforms.Select("valence"),
                transforms.Binary(5.0),
            ]
        )
        num_classes = 2
    elif args.arousal:
        label_transform = transforms.Compose(
            [
                transforms.Select("arousal"),
                transforms.Binary(5.0),
            ]
        )
        num_classes = 2
    else:
        label_transform = transforms.Compose(
            [
                transforms.Select(["valence", "arousal"]),
                transforms.Binary(5.0),
                transforms.BinariesToCategory(),
            ]
        )
        num_classes = 4

    dataset = DEAPDataset(
        root_path=args.root_path,
        io_path=f"{args.io_path}/deap",
        online_transform=online_transform,
        label_transform=label_transform,
    )

    # Model Selection
    if args.model == "EEGNet":
        model = EEGNet(
            chunk_size=chunk_size,  # default: 151
            num_electrodes=32,
            dropout=0.3,
            kernel_1=64,
            kernel_2=16,
            F1=16,
            F2=32,
            D=2,
            num_classes=num_classes,
        ).to(device)
    elif args.model == "Conformer":
        model = EEGConformer(
            n_outputs=num_classes,
            n_chans=32,
            final_fc_length=80,  # <- This is the number of features before fc, [512, 2, 40]
            pool_time_length=25,
            n_times=128,
        ).to(device)
    elif args.model == "BGRU":
        model = BGRU(num_electrodes=32, hid_channels=64, num_classes=num_classes).to(
            device
        )

    else:
        raise ValueError("Model not found")

    # Model Training and Evaluation
    k_fold = KFoldGroupbyTrial(
        n_splits=args.n_split, split_path=f"{args.io_path}/split", shuffle=True
    )

    loss_fn = nn.CrossEntropyLoss()
    test_accs, test_losses = [], []
    for fold, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        train_dataset, val_dataset = train_test_split(
            train_dataset,
            test_size=0.2,
            split_path=f"{args.io_path}/split{fold}",
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        best_val_acc = 0.0
        for epoch in range(args.epochs):
            train_loss = train(train_loader, model, loss_fn, optimizer)
            val_acc, val_loss = validate(val_loader, model, loss_fn)

            logger.info(
                f"Fold {fold}, Epoch {epoch + 1}/{args.epochs} -> Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(), f"{args.io_path}/{args.model}_fold{fold}.pt"
                )

        # Test the best model
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )
        model.load_state_dict(
            torch.load(f"{args.io_path}/{args.model}_fold{fold}.pt", weights_only=True)
        )
        test_acc, test_loss = validate(test_loader, model, loss_fn)

        logger.info(
            f"Fold {fold} -> Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}"
        )

        test_accs.append(test_acc)
        test_losses.append(test_loss)
    # Log overall results
    logger.info(
        f"Cross-Validation Results -> Average Test Accuracy: {np.mean(test_accs):.4f}, Average Test Loss: {np.mean(test_losses):.4f}"
    )


if __name__ == "__main__":
    # create environment variables
    import argparse

    parser = argparse.ArgumentParser(description="EEGNet Training")
    parser.add_argument("--seed", type=int, default=44, help="random seed")
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/dspserver/ngHAN/imsp/dataset/archive/data_preprocessed_python/",
        help="root path",
    )
    parser.add_argument(
        "--io_path", type=str, default="./io_path", help="input/output path"
    )
    parser.add_argument(
        "--log_path", type=str, default="./loggings/log", help="log path"
    )

    parser.add_argument("--model", type=str, default="EEGNet", help="model")
    parser.add_argument("--valence", action="store_true", help="valence")
    parser.add_argument("--arousal", action="store_true", help="arousal")
    parser.add_argument("--DWT", action="store_true", help="DWT")

    parser.add_argument("--n_split", type=int, default=5, help="number of splits")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

    args = parser.parse_args()
    # Setup logging

    # os.makedirs('./examples_vanilla_torch/log', exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    logger = logging.getLogger("Training")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_handler = logging.FileHandler(os.path.join(args.log_path, f"{timeticks}.log"))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    seed_everything(44)
    main(args)
