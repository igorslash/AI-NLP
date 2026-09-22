import torch  # Подключаем ядро PyTorch
import torch.nn as nn  # Подключаем инструменты для создания слоев (Linear, ReLU и т.д.)
import torch.optim as optim  # Подключаем оптимизаторы (те самые "настройщики ручек")
from torch.utils.data import Dataset, DataLoader  # Инструменты для склада и грузчика


# --- ШАГ 1: СКЛАД ДАННЫХ ---
class CustomDataset(Dataset):
    def __init__(self, data, labels):  # Добавили labels в конструктор, чтобы они были внутри
        self.data = data  # Сохраняем наши входные данные (например, картинки или векторы)
        self.labels = labels  # Сохраняем правильные ответы к этим данным

    def __len__(self):
        return len(self.data)  # Говорим PyTorch, сколько всего строк на нашем складе

    def __getitem__(self, idx):
        # Метод "дай мне один товар под номером idx"
        x = self.data[idx]  # Берем данные под этим номером
        y = self.labels[idx]  # Берем ответ под этим номером
        return x, y  # Отдаем их парой (вход, ответ)


# --- ШАГ 2: МОЗГ (НЕЙРОСЕТЬ) ---
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()  # Стандартная активация магии PyTorch
        self.flatten = nn.Flatten()  # Превращает квадратную картинку (28x28) в одну длинную линию

        # Собираем "бутерброд" из слоев
        self.linear_relu_stack = nn.Sequential(
            nn.LazyLinear(512),  # ЛЕНИВЫЙ СЛОЙ: сам поймет вход, сделает 512 выводов
            nn.ReLU(),  # НЕЙРОННЫЙ КЛЕЙ: добавляет гибкости, убирает минусы
            nn.Linear(512, 512),  # Скрытый слой: принимает 512 и отдает 512
            nn.ReLU(),  # Снова клей для сложности
            nn.Linear(512, 10),  # ФИНАЛЬНЫЙ СУДЬЯ: выдает 10 чисел (по числу классов)
        )

    def forward(self, x):
        # Описываем путь данных внутри сети
        x = self.flatten(x)  # Сначала вытягиваем данные в линию
        logits = self.linear_relu_stack(x)  # Прогоняем через все слои по очереди
        return logits  # Выдаем "сырые" ответы (логиты)


# --- ШАГ 3: ПОДГОТОВКА К ОБУЧЕНИЮ ---
# 1. Определяем, где будем считать (Видеокарта или Процессор)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Используем устройство: {device}")
model = NeuralNetwork()  # Создаем экземпляр нашей нейросети
loss_fn = nn.CrossEntropyLoss()  # Выбираем "Учителя" (считает разницу между ответом и реальностью)
optimizer = optim.SGD(model.parameters(), lr=1e-3)  # Выбираем "Механика" (крутит веса с шагом 0.001)


# 2. Создаем модель и СРАЗУ отправляем её на это устройство
model = NeuralNetwork().to(device)

# 3. Создаем "пустую" картинку 28x28 прямо на этом устройстве
# (1, 28, 28) означает: 1 картинка размером 28 на 28 пикселей
X = torch.rand(1, 28, 28, device=device)

# 4. Просим модель "взглянуть" на эту картинку
logits = model(X)

# 5. Превращаем странные числа в красивые вероятности (0.0 ... 1.0)
# dim=1 означает, что сумма всех 10 ответов будет равна 1.0 (100%)
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Узнаем, какое число модель считает самым вероятным
y_pred = pred_probab.argmax(1)
print(f"Предсказанный класс: {y_pred.item()}")

# --- ШАГ 4: САМО ОБУЧЕНИЕ ---
for epoch in range(10):  # Повторяем весь процесс 10 раз (10 эпох)
    # Перебираем батчи, которые нам привез "Грузчик" (DataLoader)
    for batch, (X, y) in enumerate(train_loader):
        # 1. ПРЕДСКАЗАНИЕ
        pred = model(X)  # Просим модель угадать, что на картинках в батче

        # 2. ОЦЕНКА ОШИБКИ
        loss = loss_fn(pred, y)  # Учитель сравнивает предсказание (pred) с правдой (y)

        # 3. ПОДГОТОВКА (ОБНУЛЕНИЕ)
        optimizer.zero_grad()  # Забываем старые ошибки, чтобы не путаться

        # 4. ОТКАТКА (BACKPROPAGATION)
        loss.backward()  # Идем назад от ошибки и считаем, какие веса виноваты больше всего

        # 5. КОРРЕКЦИЯ (ШАГ)
        optimizer.step()  # Механик реально подкручивает веса в нужную сторону

        # ЛОГИРОВАНИЕ (чтобы мы не скучали)
        if batch % 100 == 0:  # Каждые 100 батчей выводим статус
            loss_val = loss.item()  # Берем числовое значение ошибки
            current = batch * len(X)  # Считаем, сколько картинок уже прошли
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")  # Печатаем лосс и прогресс

