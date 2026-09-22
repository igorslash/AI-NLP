from transformers import (AutoModelForSequenceClassification,
                          AdamW, get_scheduler)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=2)
# Оптимизатор - наш "механик"
optimizer = AdamW(model.parameters(), lr=5e-5)

# Планировщик - плавно снижает скорость обучения к финалу
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0,
    num_training_steps=num_training_steps
)
