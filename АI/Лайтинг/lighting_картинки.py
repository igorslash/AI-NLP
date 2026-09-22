import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl


# ─────────────────────────────────────────────────────────────
# 1. МОДЕЛЬ (LightningModule)
# ─────────────────────────────────────────────────────────────
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# ─────────────────────────────────────────────────────────────
# 2. ДАННЫЕ (LightningDataModule)
# ─────────────────────────────────────────────────────────────
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, data_dir="./data"):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = ToTensor()

    def prepare_data(self):
        # Скачиваем 1 раз (не сохраняем в self!)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Создаём датасеты для каждого процесса
        self.train_dataset = MNIST(self.data_dir, train=True, download=False, transform=self.transform)
        self.val_dataset = MNIST(self.data_dir, train=False, download=False, transform=self.transform)
        self.test_dataset = MNIST(self.data_dir, train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


# ─────────────────────────────────────────────────────────────
# 3. ЗАПУСК (Trainer)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Инициализация
    model = LitModel()
    dm = MNISTDataModule(batch_size=32)

    # Создаём Trainer
    trainer = pl.Trainer(
        max_epochs=5,           # Сколько эпох обучать
        accelerator='auto',     # Автоматически выберет CPU или GPU
        devices='auto',         # Автоматически выберет доступные устройства
        log_every_n_steps=10    # Как часто логировать метрики
    )

    # 🚀 ЗАПУСК ОБУЧЕНИЯ
    trainer.fit(model, dm)

    # ✅ ЗАПУСК ТЕСТА (после обучения)
    trainer.test(model, dm)