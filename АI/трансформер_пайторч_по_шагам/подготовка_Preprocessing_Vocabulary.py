from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Функция-"ножницы": только режет (truncation) и превращает в ID
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"],
                     truncation=True)

