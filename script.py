import json

config = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "vocab_size": 32000,
    "model_type": "llama"
}

config_file_path = '/home/haridoss/Models/llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct/config.json'

with open(config_file_path, 'w') as config_file:
    json.dump(config, config_file, indent=4)

print(f"Configuration file saved to {config_file_path}")