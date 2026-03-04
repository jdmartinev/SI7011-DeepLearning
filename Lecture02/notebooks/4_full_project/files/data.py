# data.py
# ───────────────────────────────────────────────────────────────
# LightningDataModule for MNIST.
# Responsible for: downloading, splitting, transforming, and
# returning DataLoaders. Nothing else lives here.
# ───────────────────────────────────────────────────────────────

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    """
    Self-contained data pipeline for MNIST.

    Args:
        data_dir    : path to download/cache the dataset
        batch_size  : samples per batch
        num_workers : DataLoader worker processes (0 = main process)
        val_split   : fraction of training data reserved for validation
    """

    def __init__(self, data_dir="./data", batch_size=64, num_workers=2, val_split=0.1):
        super().__init__()
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.val_split   = val_split

        # Normalize using pre-computed MNIST mean and std
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ])

    def prepare_data(self):
        """
        Download data to disk. Called once on a single process.
        Do NOT assign state here (no self.x = ...) — this method
        is not called on every GPU in multi-GPU setups.
        """
        datasets.MNIST(self.data_dir, train=True,  download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Load data into memory and apply splits/transforms.
        Called on every GPU process — safe to assign self.x here.

        stage: 'fit' | 'validate' | 'test' | 'predict' | None
        """
        if stage in ("fit", None):
            full_train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            val_size   = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size
            self.mnist_train, self.mnist_val = random_split(full_train, [train_size, val_size])

        if stage in ("test", None):
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,           # Always shuffle training data
            num_workers=self.num_workers,
            pin_memory=True,        # Faster CPU→GPU transfers
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,          # Never shuffle val/test
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
