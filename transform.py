import torch
from torch.utils.data import DataLoader, Subset
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from braindecode.models import EEGNetv1
from braindecode.util import set_random_seeds
import lightning as L
from torchmetrics.functional import accuracy

# ---- DATASET CONFIGURATION ---- #
root_path = '/mnt/inaisfs/user-fs/imsp/Han/dataset/DEAP/data_preprocessed_python/'
dataset = DEAPDataset(
    io_path='./phase1/deap',
    root_path=root_path,
    offline_transform=transforms.Compose([
        transforms.BandDifferentialEntropy(apply_to_baseline=True),
        transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True)
    ]),
    online_transform=transforms.Compose([
        transforms.BaselineRemoval(),
        transforms.ToTensor()
    ]),
    label_transform=transforms.Compose([
        transforms.Select(['valence', 'arousal']),
        transforms.Binary(5.0),
        transforms.BinariesToCategory()
    ]),
    num_worker=8,
    chunk_size=512  # Increased chunk size to accommodate EEGNetv1's requirements
)

dataset = DEAPDataset(root_path=root_path,
                      io_path='./phase1/deap',
                      online_transform=transforms.Compose([
                          transforms.To2d(),
                          transforms.ToTensor()
                      ]),
                      label_transform=transforms.Compose([
                          transforms.Select(['valence', 'arousal']),
                          transforms.Binary(5.0),
                          transforms.BinariesToCategory()
                      ]))

# ---- K-FOLD CROSS VALIDATION ---- #
k_fold = KFoldGroupbyTrial(n_splits=5, split_path='./phase1/split', shuffle=True, random_state=44)
splits = k_fold.split(dataset)

# ---- MODEL CONFIGURATION ---- #
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = False  # Set to False for reproducibility

seed = 44
set_random_seeds(seed=seed, cuda=cuda)

# Extract dataset dimensions from a sample

sample_data = dataset[0][0]
n_channels = sample_data.shape[1]  # Changed to account for the reshaped input
n_times = sample_data.shape[2]     # Changed to account for the reshaped input
print(f'Number of channels: {n_channels}')
print(f'Number of time steps: {n_times}')

# Define EEGNetv1 with correct dimensions
model = EEGNetv1(
    n_chans=n_channels,
    n_outputs=4,
    n_times=n_times,
    final_conv_length="auto",
).to(device)

# ---- LIGHTNING MODULE ---- #
class LitModule(L.LightningModule):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch['eeg'], batch['label']
        y_hat = self.module(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['eeg'], batch['label']
        y_hat = self.module(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y, "multiclass", num_classes=4)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=0.000625, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2 - 1)
        return [optimizer], [scheduler]

# ---- MAIN EXECUTION ---- #
if __name__ == '__main__':
    for fold, (train_indices, test_indices) in enumerate(splits):
        import pdb; pdb.set_trace()
        print(f"Processing Fold {fold + 1}...")

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        # Debug information
        for batch in train_loader:
            print(f"Train batch EEG shape: {batch['eeg'].shape}, Label shape: {batch['label'].shape}")
            break
        
        trainer = L.Trainer(
            max_epochs=2,
            accelerator="gpu" if cuda else "cpu",
            devices=1,
            deterministic=True  # Added for reproducibility
        )
        lit_model = LitModule(model)
        trainer.fit(lit_model, train_loader)
        trainer.test(lit_model, dataloaders=test_loader)