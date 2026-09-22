from transformers import (TrainingArguments,
                          AutoModelForSequenceClassification)

# Все настройки (LR, Epochs, Weight Decay) в одной куче:
training_args = TrainingArguments(
    output_dir="test-trainer",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch" # Когда проверять качество
)

model = (AutoModelForSequenceClassification
         .from_pretrained("bert-base-uncased", num_labels=2))

