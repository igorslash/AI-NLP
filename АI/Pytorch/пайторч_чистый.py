import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Исправленный Dataset (теперь он принимает реальные данные)
class MyDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.x = data_tensor
        self.y = labels_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
# 2. Модель (Простая нейронка с ReLU)
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- НАСТРОЙКИ ---
input_dim = 10    # Размер входного вектора (например, 10 признаков)
hidden_dim = 32   # Количество нейронов в скрытом слое
output_dim = 2    # Количество классов (например, 0 или 1)
num_epochs = 5
batch_size = 4

# Определяем устройство (Для твоего Мака - mps, для PC - cuda)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Используем устройство: {device}")

# 3. СОЗДАЕМ МОДЕЛЬ (Сначала модель!)
model = MyModel(input_dim, hidden_dim, output_dim).to(device)

# 4. НАСТРОЙКИ ОБУЧЕНИЯ (Теперь оптимизатор видит параметры модели)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Генерируем "рыбу" (фейковые данные) для теста
mock_data = torch.randn(100, input_dim) 
mock_labels = torch.randint(0, output_dim, (100,))
dataset = MyDataset(mock_data, mock_labels)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. ЦИКЛ ОБУЧЕНИЯ (The Hard Way)
for epoch in range(num_epochs):
    model.train() # Переводим в режим обучения
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        # Переносим данные на M4 или GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Обнуляем градиенты (ОБЯЗАТЕЛЬНО!)
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Обратный проход (магия градиентов)
        loss.backward()
        
        # Обновление весов
        optimizer.step()

        running_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f'Эпоха [{epoch + 1}/{num_epochs}], Шаг [{i + 1}/{len(train_loader)}], Ошибка: {loss.item():.4f}')

print("Обучение завершено!")
