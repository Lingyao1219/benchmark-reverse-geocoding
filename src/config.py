# LLM Configuration
MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# o3
# gpt-4.1
# gpt-4.1-mini
# gpt-4o
# gpt-4o-mini
# claude-3-7-sonnet-20250219
# claude-3-5-haiku-20241022
# Qwen/Qwen2.5-VL-3B-Instruct ### Double check
# Qwen/Qwen2.5-VL-7B-Instruct ### Double check
# Qwen/Qwen2.5-VL-32B-Instruct ### Double check
# Qwen/Qwen2.5-VL-72B-Instruct
# meta-llama/Llama-3.1-30B-Vision-Instruct-Turbo ### Double check
# meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo
# meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
# meta-llama/Llama-4-Scout-17B-16E-Instruct

### Gemini pro


DEFAULT_PROVIDER = "togetherai" 
# openai, anthropic, togetherai

# LLM Query Parameters
TEMPERATURE = 0.2

# File Configuration
IMAGE_INPUT_FOLDER = "images"
RESULTS_OUTPUT_FILE = "image_results.jsonl"

# API Key Management
SECRETS_FILE_PATH = "secrets.txt"