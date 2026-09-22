import lightning as L
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd  # для загрузки CSV

# ===== DATASET =====
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ===== DATAMODULE =====
class MyDataModule(L.LightningDataModule):
    def __init__(self, train_csv, val_csv, tokenizer_name="bert-base-uncased", batch_size=32):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Загружаем данные
        train_df = pd.read_csv(self.train_csv)
        val_df = pd.read_csv(self.val_csv)
        
        # Предполагаем колонки: 'text' и 'label'
        self.train_dataset = TextDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer
        )
        self.val_dataset = TextDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# ===== MODEL =====
class MyModel(L.LightningModule):
    def __init__(self, model_name="bert-base-uncased", num_labels=2, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.lr = lr
    
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == batch["labels"]).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# ===== ЗАПУСК =====
if __name__ == "__main__":
    # Создаём DataModule
    dm = MyDataModule(
        train_csv="train.csv",
        val_csv="val.csv",
        batch_size=16
    )
    
    # Создаём модель
    model = MyModel(num_labels=2)
    
    # Callbacks
    checkpoint = L.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min"
    )
    early_stop = L.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    
    # Trainer
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=3,
        callbacks=[checkpoint, early_stop],
        log_every_n_steps=10
    )
    
    # Обучение
    trainer.fit(model, dm)
    
    # Сохраняем лучшую модель
    trainer.save_checkpoint("final_model.ckpt")
