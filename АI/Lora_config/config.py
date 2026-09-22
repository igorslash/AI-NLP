import torch
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import re


class LoRADetector:
    """
    Универсальный детектор слоёв для LoRA.
    Работает с любой моделью из Hugging Face.
    """

    def __init__(self, model_name):
        print(f"🔍 Анализирую модель: {model_name}")
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.attention_layers = self._find_attention_layers()

    def _find_attention_layers(self):
        """Ищет все возможные слои внимания в модели"""
        attention_layers = {
            'query': [],
            'key': [],
            'value': [],
            'output': [],
            'all': []
        }

        # Ключевые слова для разных типов слоёв (на всех языках)
        query_patterns = ['q_', 'query', 'qproj', 'q_proj']
        key_patterns = ['k_', 'key', 'kproj', 'k_proj']
        value_patterns = ['v_', 'value', 'vproj', 'v_proj']
        output_patterns = ['o_', 'out', 'output', 'oproj', 'o_proj', 'dense']

        for name, module in self.model.named_modules():
            name_lower = name.lower()

            # Проверяем, что это вообще слой внимания
            if not any(x in name_lower for x in ['attention', 'attn', 'self']):
                continue

            # Запоминаем все найденные слои
            attention_layers['all'].append(name)

            # Проверяем Query слои
            if any(p in name_lower for p in query_patterns):
                attention_layers['query'].append(name)

            # Проверяем Key слои
            if any(p in name_lower for p in key_patterns):
                attention_layers['key'].append(name)

            # Проверяем Value слои
            if any(p in name_lower for p in value_patterns):
                attention_layers['value'].append(name)

            # Проверяем Output слои
            if any(p in name_lower for p in output_patterns):
                attention_layers['output'].append(name)

        return attention_layers

    def suggest_lora_config(self, strategy='q+v'):
        """
        Предлагает конфигурацию LoRA на основе стратегии:
        - 'q+v' : только Query и Value (классика)
        - 'all' : все слои внимания
        - 'qkvo' : Query, Key, Value, Output
        - 'auto' : автоматический выбор
        """

        target_modules = []

        if strategy == 'q+v':
            target_modules = self.attention_layers['query'] + self.attention_layers['value']
        elif strategy == 'qkvo':
            target_modules = (self.attention_layers['query'] +
                              self.attention_layers['key'] +
                              self.attention_layers['value'] +
                              self.attention_layers['output'])
        elif strategy == 'all':
            target_modules = self.attention_layers['all']
        else:  # auto
            # Умный выбор: если есть q_proj/v_proj — берём их
            if self.attention_layers['query'] and self.attention_layers['value']:
                target_modules = self.attention_layers['query'] + self.attention_layers['value']
            else:
                # Иначе берём первые 4 слоя из всех
                target_modules = self.attention_layers['all'][:4]

        # Убираем дубликаты
        target_modules = list(set(target_modules))

        return {
            'model_type': self.model.config.model_type,
            'found_layers': {
                'query': self.attention_layers['query'][:3],  # покажем первые 3
                'key': self.attention_layers['key'][:3],
                'value': self.attention_layers['value'][:3],
                'output': self.attention_layers['output'][:3],
                'total_attention': len(self.attention_layers['all'])
            },
            'suggested_modules': target_modules,
            'config': LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
            )
        }

    def print_summary(self):
        """Красиво печатает найденные слои"""
        print("\n" + "=" * 50)
        print(f"📊 Анализ модели: {self.model_name}")
        print("=" * 50)

        print(f"\n🔹 Тип модели: {self.model.config.model_type}")
        print(f"🔹 Всего слоёв внимания: {len(self.attention_layers['all'])}")

        print("\n🔹 Найденные слои:")
        for layer_type in ['query', 'key', 'value', 'output']:
            if self.attention_layers[layer_type]:
                print(f"  • {layer_type.upper()}: {len(self.attention_layers[layer_type])} слоёв")
                print(f"    Пример: {self.attention_layers[layer_type][0]}")

        print("\n💡 Рекомендации:")
        print("  • Для дообучения обычно используют Query и Value слои")
        print("  • Можно также добавить Key и Output для сложных задач")

        print("\n📋 Пример конфигурации LoRA:")
        print('```python')
        print('from peft import LoraConfig')
        print()
        print('config = LoraConfig(')
        print('    r=8,')
        print('    lora_alpha=32,')
        print(f'    target_modules={self.attention_layers["query"][:2] + self.attention_layers["value"][:2]},')
        print('    lora_dropout=0.1,')
        print('    bias="none",')
        print(')')
        print('```')


# ===== ИСПОЛЬЗОВАНИЕ =====

# 1. Для любой модели одним вызовом
detector = LoRADetector("mistralai/Mistral-7B-v0.1")
detector.print_summary()

# 2. Получить готовую конфигурацию
suggestion = detector.suggest_lora_config(strategy='q+v')
config = suggestion['config']
print(f"✅ Использую слои: {suggestion['suggested_modules']}")

# 3. Сразу применить к модели
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = get_peft_model(model, config)
model.print_trainable_parameters()


def auto_lora_config(model_name, strategy='q+v'):
    """
    Создаёт LoRA конфиг автоматически для любой модели
    """
    detector = LoRADetector(model_name)
    suggestion = detector.suggest_lora_config(strategy)
    return suggestion['config']

# Использование одной строкой!
config = auto_lora_config("mistralai/Mistral-7B-v0.1")
model = get_peft_model(model, config)