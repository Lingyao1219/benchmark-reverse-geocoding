# API Key Management
SECRETS_FILE_PATH = "secrets.txt"

# LLM Configuration
MODEL = "gpt-4.1-mini"
DEFAULT_PROVIDER = "openai" # openai, anthropic, togetherai

# LLM Query Parameters
TEMPERATURE = 0.7

# File Configuration
DATASET = "test_dataset22"

# Save Filename
if '/' in MODEL:
    MODEL_FILENAME = MODEL.split('/')[-1]
else:
    MODEL_FILENAME = MODEL

IMAGE_INPUT_FOLDER = f"data/{DATASET}"
RESULTS_OUTPUT_FILE = f"result/{DATASET}_{MODEL_FILENAME}.jsonl"

# o3
# gpt-4.1
# gpt-4.1-mini
# gpt-4o
# gpt-4o-mini
# claude-3-7-sonnet-20250219
# claude-3-5-haiku-20241022
# Qwen/Qwen2.5-7B-Instruct-Turbo
# Qwen/QwQ-32B
# Qwen/Qwen2.5-VL-72B-Instruct
# meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo
# meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo
# meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
# meta-llama/Llama-4-Scout-17B-16E-Instruct

# "o3": (2, 8),
# "o3-mini": (1.1, 4.4),
# "o4-mini": (1.1, 4.4),
# "gpt-4.1": (2.0, 8.0),
# "gpt-4.1-mini": (0.4, 1.6), 
# "gpt-4.1-nano": (0.1, 0.4), 
# "claude-3-5-haiku-20241022": (0.8, 4.0),
# "claude-3-7-sonnet-20250219": (3.0, 15.0),
# "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
# "Qwen/QwQ-32B": (1.2, 1.2),
# "Qwen/Qwen2.5-VL-72B-Instruct": (1.20, 1.20),
# "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": (0.18, 0.18),
# "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": (1.20, 1.20),
# "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
# "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),