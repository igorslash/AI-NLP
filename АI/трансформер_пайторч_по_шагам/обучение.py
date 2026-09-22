from accelerate import Accelerator
from tqdm.auto import tqdm

accelerator = Accelerator()
# Магия Accelerator: сам перекинет всё на GPU/MPS
train_dl, model, optimizer = accelerator.prepare(train_dataloader,
                                                 model, optimizer)

progress_bar = tqdm(range(num_training_steps))

model.train()  # Режим "Учеба"
for epoch in range(num_epochs):
    for batch in train_dl:
        # Проход вперед: получаем логиты и Loss
        outputs = model(**batch)
        loss = outputs.loss

        # Проход назад: вычисляем ошибки
        accelerator.backward(loss)

        # Подкручиваем веса
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()  # Стираем старые ошибки!

        progress_bar.update(1)
