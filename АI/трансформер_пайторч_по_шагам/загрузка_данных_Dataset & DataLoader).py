from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
# Массовая нарезка
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Оставляем только тензоры для PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids",
                                                "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Конвейер (лоадер), который выдает готовые батчи
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator
)
