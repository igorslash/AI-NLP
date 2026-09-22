from transformers import DataCollatorWithPadding

# Этот парень сам сделает паддинг внутри батчей во время обучения
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



