import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# --- 1. ДАННЫЕ (Перенесли наверх) ---
checkpoint = "bert-base-uncased"
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



# --- 2. ЛОГИКА СОЗДАНИЯ МОДЕЛИ (для model_init) ---
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# --- 3. НАСТРОЙКИ ---
trainer_arguments = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch", # Должно совпадать с eval_strategy для load_best_model
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True, # Оставит в памяти лучшую версию модели
    fp16 = True,
)

# --- 4. ТРЕНЕР ---
trainer = Trainer(
    model_init=model_init, # Используем функцию инициализации
    args=trainer_arguments,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    # Мы пока не определили функцию compute_metrics, можно временно убрать или дописать
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# --- 5. ЗАПУСК ---
trainer.train()
trainer.evaluate()
trainer.save_model("./my_best_model")