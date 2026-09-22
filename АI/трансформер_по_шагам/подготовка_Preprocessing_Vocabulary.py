from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



def tokenize_fn(batch):
    # Только нарезка (truncation), без паддинга!
    return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)


