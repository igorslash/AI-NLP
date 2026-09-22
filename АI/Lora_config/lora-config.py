# Для LLaMA, Mistral, Falcon
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Для BERT, RoBERTa
config = LoraConfig(
    r=8,
    target_modules=["query", "value"],
    task_type="SEQ_CLS"
)

# Для T5
config = LoraConfig(
    r=8,
    target_modules=["q", "v"],
    task_type="SEQ_2_SEQ_LM"
)
#-------------------------------------
config = LoraConfig(
    target_modules="all-linear", # Он САМ найдет все Q, K, V и Dense слои
    task_type="SEQ_2_SEQ_LM"
)
