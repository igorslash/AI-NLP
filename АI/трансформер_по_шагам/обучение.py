from transformers import Trainer

# Собираем оркестр
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer, # Чтобы модель умела сама сохраняться
)

# ПОЕХАЛИ! (Внутри этой строки спрятан весь тот код на PyTorch)
trainer.train()

