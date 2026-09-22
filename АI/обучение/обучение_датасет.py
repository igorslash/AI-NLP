from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Загружаем реальный датасет (например, код на Python)
raw_datasets = load_dataset("code_search_net", "python", split="train", streaming=True)

# 2. Создаем "умный" итератор (генератор)
# Он будет выдавать тексты порциями по 1000, чтобы не "взорвать" оперативку
def get_training_corpus():
    dataset_iter = iter(raw_datasets)
    while True:
        # Берем следующие 1000 примеров
        batch = [next(dataset_iter)["whole_func_string"] for _ in range(1000)]
        yield batch

training_corpus = get_training_corpus()

# 3. БАЗА: Берем старый токенизатор как шаблон
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 4. ОБУЧЕНИЕ: Вот здесь происходит магия пересчета статистики!
# Мы просим его выучить 52000 самых частых кусков кода из нашего итератора
new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=52000)

# 5. ПРОВЕРКА
example = "def my_function(self, x):"
print("Как нарезал новый:", new_tokenizer.tokenize(example))
