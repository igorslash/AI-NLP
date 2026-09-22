import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# ─────────────────────────────────────────────────────────────
# 1. МОДЕЛЬ (LightningModule) - Твой Transformer из 4 главы
# ─────────────────────────────────────────────────────────────
class LitTextModel(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        # Вместо Flatten() для картинок — Embedding для текста
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Здесь мог бы быть твой TransformerBlock из книги Рашки
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        
        self.fc = nn.Linear(embed_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: [batch_size, seq_len] -> заходят ID токенов
        x = self.embedding(x)   # [batch_size, seq_len, embed_dim]
        x = self.transformer(x) 
        x = x.mean(dim=1)       # Схлопываем по длине текста (Global Average Pooling)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # sync_dist=True запускает all_reduce по всем GPU перед логированием
        self.log(self.log('train_loss', loss, sync_dist=True, prog_bar=True))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

# ─────────────────────────────────────────────────────────────
# 2. ДАННЫЕ (LightningDataModule) - Токенизация вместо ToTensor()
# ─────────────────────────────────────────────────────────────
class TextDataModule(pl.LightningDataModule):
    def __init__(self, texts, labels, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.texts = texts # Список списков с ID токенов (уже после токенизатора)
        self.labels = labels

    def setup(self, stage=None):
        # Создаем простой датасет (как в 5 главе Рашки)
        class SimpleDataset(Dataset):
            def __init__(self, x, y):
                self.x = torch.tensor(x, dtype=torch.long)
                self.y = torch.tensor(y, dtype=torch.long)
            def __len__(self): return len(self.x)
            def __getitem__(self, idx): return self.x[idx], self.y[idx]

        self.train_ds = SimpleDataset(self.texts, self.labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

# ─────────────────────────────────────────────────────────────
# 3. ЗАПУСК
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Представь, что мы уже токенизировали текст: 100 примеров по 10 слов
    mock_data = [[1, 5, 2, 8, 3, 0, 0, 0, 0, 0]] * 100 
    mock_labels = [1] * 50 + [0] * 50

    dm = TextDataModule(mock_data, mock_labels)
    model = LitTextModel(vocab_size=1000, embed_dim=64, num_classes=2)

    trainer = pl.Trainer(max_epochs=3, accelerator='auto')
    trainer.fit(model, dm)
