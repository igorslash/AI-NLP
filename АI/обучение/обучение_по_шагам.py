import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import lightning as L
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 1. ПОДГОТОВКА (Preprocessing / Vocabulary)
# ==========================================
# В реальности тут будет токенизатор, но для примера:
vocab = {"<PAD>": 0, "привет": 1, "мир": 2, "pytorch": 3, "крутой": 4}
data_x = [[1, 2], [3, 4], [1, 3]]  # "привет мир", "pytorch крутой", "привет pytorch"
data_y = [1, 0, 1]  # Метки: 1 - позитив, 0 - нейтрал

#config = {
   # "v_size": 30000,
   # "e_dim": 256,
   # "classes": 3
#}
# Создание модели становится простым:
model = NLPClassifier(config["v_size"], 
config["e_dim"], config["classes"])



# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ (Dataset & DataLoader)
# ==========================================
class SimpleNLPDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [torch.tensor(t) for t in texts]
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


dataset = SimpleNLPDataset(data_x, data_y)
# DataLoader собирает батчи по 2 примера
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)


# ==========================================
# 3. АРХИТЕКТУРА МОДЕЛИ (The Architecture)
# ==========================================
class NLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        # Слой вложений (превращает ID слова в вектор смысла)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Линейный слой (голова классификатора)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        # Агрегация: усредняем смыслы всех слов в предложении
        pooled = embedded.mean(dim=1)  # [batch, embed_dim]
        # Логиты на выходе
        return self.fc(pooled)  # [batch, num_classes]


# ==========================================
# 4. МЕХАНИКА ОБУЧЕНИЯ (Loss & Optimizer)
# ==========================================
model = NLPClassifier(vocab_size=len(vocab), embed_dim=8, num_classes=2)
criterion = nn.CrossEntropyLoss()  # Судья
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Тренер

# ==========================================
# 5. ЦИКЛ ОБУЧЕНИЯ (Training Loop)
# ==========================================
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        # А) Обнуляем
        optimizer.zero_grad()

        # Б) Прогоняем (Forward)
        logits = model(batch_x)

        # В) Считаем ошибку (Loss)
        loss = criterion(logits, batch_y)

        # Г) Считаем градиенты (Backward)
        loss.backward()

        # Д) Обновляем веса (Step)
        optimizer.step()

    if epoch % 2 == 0:
        print(f"Эпоха {epoch}, Ошибка: {loss.item():.4f}")

print("✅ Обучение завершено!")

#----------------------------------------------------------------


class SummarizationModule(L.LightingModule):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def predict_step(self, batch):
        summary_ids = self.model.generate(batch["input_ids"],
                                          attention_mask=batch["attention_mask"], max_length=128)
        return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
texts = ["Спустя почти 10 лет ухожу от Tele2. Предложили невыгодный тариф...", "Второй текст..."]
tokenized_data = sum_tokenizer(texts, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
loader = L.DataLoader(tokenized_data)
lit_model = SummarizationModule("facebook/bart-large-cnn")
trainer = L.Trainer(accelerator="auto", devices=1)
result = trainer.predict(lit_model, loader)
for batch in result:
    print(batch)


