# Для разных типов моделей
models = {
    "LLaMA": "meta-llama/Llama-2-7b-hf",
    "Mistral": "mistralai/Mistral-7B-v0.1",
    "BERT": "bert-base-uncased",
    "GPT2": "gpt2",
    "T5": "t5-base",
    "ViT": "google/vit-base-patch16-224"
}

# Проверяем любую модель за секунду
for name, model_path in models.items():
    print(f"\n=== {name} ===")
    detector = LoRADetector(model_path)
    detector.print_summary()
    print("\n" + "="*30)