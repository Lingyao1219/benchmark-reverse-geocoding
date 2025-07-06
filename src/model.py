import json
import re
import ast
from typing import Optional, Dict, Any, List, Tuple
from openai import OpenAI
import anthropic
import config

# Cost tracking configuration (prices per 1M tokens)
MODEL_COSTS = {
    # OpenAI Models
    "o3": (2.0, 8.0),
    "o3-mini": (1.1, 4.4),
    "o4-mini": (1.1, 4.4),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (5.0, 20.0),
    "gpt-4o-mini": (0.60, 2.4),
    
    # Anthropic Models
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    "claude-3-7-sonnet-20250219": (3.0, 15.0),
    
    # Together AI Models
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
    "Qwen/QwQ-32B": (1.2, 1.2),
    "Qwen/Qwen2.5-VL-72B-Instruct": (1.20, 1.20),
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": (0.18, 0.18),
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": (1.20, 1.20),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
}

def estimate_tokens(text: str) -> int:
    """Rough estimation: ~4 characters per token for most models"""
    return len(text) // 4

def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost based on token usage and model pricing"""
    if model not in MODEL_COSTS:
        print(f"Warning: No pricing data for model {model}")
        return 0.0
    
    input_cost_per_1m, output_cost_per_1m = MODEL_COSTS[model]
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    
    return input_cost + output_cost

def load_api_key(provider: str) -> str:
    provider_key_map = {
        'openai': 'openai_key',
        'anthropic': 'claude_key',
        'togetherai': 'together_key'
    }
    key_identifier = provider_key_map.get(provider.lower())
    if not key_identifier:
        raise ValueError(f"Unknown provider: {provider}")
    try:
        with open(config.SECRETS_FILE_PATH) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[0].strip() == key_identifier:
                    return parts[1].strip()
        raise ValueError(f"{key_identifier} not found in {config.SECRETS_FILE_PATH}")
    except FileNotFoundError:
        raise ValueError(f"Secrets file not found: {config.SECRETS_FILE_PATH}")

def query_openai(
    messages_payload: List[Dict[str, Any]],
    model: str
) -> Tuple[str, Dict[str, Any]]:
    api_key = load_api_key(provider='openai')
    client = OpenAI(api_key=api_key)

    response_format_param = {"type": "json_object"}

    if any(m in model for m in ["o3", "o3-mini"]):
        response = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            response_format=response_format_param
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            temperature=config.TEMPERATURE,
            response_format=response_format_param
        )
    
    # Extract usage info
    usage_info = {
        'input_tokens': response.usage.prompt_tokens if response.usage else 0,
        'output_tokens': response.usage.completion_tokens if response.usage else 0,
        'total_tokens': response.usage.total_tokens if response.usage else 0
    }
    
    return response.choices[0].message.content, usage_info

def query_claude(
    system_prompt: str,
    user_content_blocks: List[Dict[str, Any]],
    model: str,
    temperature: float
) -> Tuple[str, Dict[str, Any]]:
    api_key = load_api_key(provider='anthropic')
    client = anthropic.Anthropic(api_key=api_key)
    anthropic_messages = [{"role": "user", "content": user_content_blocks}]

    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=anthropic_messages,
        temperature=temperature,
        max_tokens=5000
    )
    
    # Extract usage info
    usage_info = {
        'input_tokens': response.usage.input_tokens,
        'output_tokens': response.usage.output_tokens,
        'total_tokens': response.usage.input_tokens + response.usage.output_tokens
    }
    
    for item in response.content:
        if item.type == 'text':
            return item.text, usage_info
    return "", usage_info

def query_togetherai(
    messages_payload: List[Dict[str, Any]],
    model: str,
    temperature: float
) -> Tuple[str, Dict[str, Any]]:
    api_key = load_api_key(provider='togetherai')
    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    response = client.chat.completions.create(
        model=model,
        messages=messages_payload,
        temperature=temperature,
    )
    
    usage_info = {
        'input_tokens': response.usage.prompt_tokens if response.usage else 0,
        'output_tokens': response.usage.completion_tokens if response.usage else 0,
        'total_tokens': response.usage.total_tokens if response.usage else 0
    }
    
    return response.choices[0].message.content, usage_info

def query_llm(
    text_prompt: str,
    system_prompt: Optional[str] = None,
    image_base64_data: Optional[str] = None,
    image_mime_type: Optional[str] = None,
    model: Optional[str] = config.MODEL,
    provider: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    model = model or config.MODEL

    common_system_prompt = system_prompt or "You are a helpful AI assistant."
    if "json" in text_prompt.lower() and "JSON" not in common_system_prompt:
        common_system_prompt += " Ensure your response is a valid JSON object."

    if provider == "openai":
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        if image_base64_data and image_mime_type:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime_type};base64,{image_base64_data}"}
            })
        messages_payload: List[Dict[str, Any]] = []
        if common_system_prompt:
            messages_payload.append({"role": "system", "content": common_system_prompt})
        messages_payload.append({"role": "user", "content": user_content})
        
        response_text, usage_info = query_openai(messages_payload, model)
        
    elif provider == "anthropic":
        user_content_blocks_list: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        if image_base64_data and image_mime_type:
            user_content_blocks_list.insert(0, {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_mime_type,
                    "data": image_base64_data,
                },
            })
        response_text, usage_info = query_claude(common_system_prompt, user_content_blocks_list, model, config.TEMPERATURE)
        
    elif provider == "togetherai":
        user_content_tg: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
        if image_base64_data and image_mime_type:
            user_content_tg.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime_type};base64,{image_base64_data}"}
            })
        messages_payload_tg: List[Dict[str, Any]] = []
        if common_system_prompt:
            messages_payload_tg.append({"role": "system", "content": common_system_prompt})
        messages_payload_tg.append({"role": "user", "content": user_content_tg})
        
        response_text, usage_info = query_togetherai(messages_payload_tg, model, config.TEMPERATURE)
        
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Calculate cost
    cost = calculate_cost(usage_info['input_tokens'], usage_info['output_tokens'], model)
    usage_info['cost_usd'] = cost
    usage_info['model'] = model
    
    return response_text, usage_info


def parse_json(response_text: Optional[str], default_value: Optional[Dict] = None) -> Dict[str, Any]:
    if default_value is None:
        default_value = {}
    if not response_text:
        return default_value

    text_to_parse = response_text.strip()

    match_json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_to_parse, re.DOTALL)
    if match_json_block:
        text_to_parse = match_json_block.group(1).strip()

    try:
        return json.loads(text_to_parse)
    except json.JSONDecodeError:
        pass

    cleaned_text = text_to_parse.replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    try:
        evaluated = ast.literal_eval(cleaned_text)
        if isinstance(evaluated, (dict, list)):
            return json.loads(json.dumps(evaluated))
    except (ValueError, SyntaxError, TypeError, MemoryError): 
        pass 

    if '{' in cleaned_text and '}' in cleaned_text:
        temp_cleaned_text = re.sub(r'(?<=[{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', cleaned_text)
        if "'" in temp_cleaned_text:
            temp_cleaned_text = temp_cleaned_text.replace("'", '"')
        
        temp_cleaned_text = re.sub(r',\s*([}\]])', r'\1', temp_cleaned_text)
        try:
            return json.loads(temp_cleaned_text)
        except json.JSONDecodeError:
            pass

    if not match_json_block: 
        start_index, end_index = -1, -1
        first_brace, first_bracket = text_to_parse.find('{'), text_to_parse.find('[')

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_index, end_index = first_brace, text_to_parse.rfind('}')
        elif first_bracket != -1:
            start_index, end_index = first_bracket, text_to_parse.rfind(']')

        if start_index != -1 and end_index > start_index:
            potential_json = text_to_parse[start_index : end_index+1]
            try:
                cleaned_potential_json = potential_json.replace("None", "null").replace("True", "true").replace("False", "false")
                cleaned_potential_json = re.sub(r'(?<=[{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', cleaned_potential_json)
                if "'" in cleaned_potential_json:
                    cleaned_potential_json = cleaned_potential_json.replace("'", '"')
                cleaned_potential_json = re.sub(r',\s*([}\]])', r'\1', cleaned_potential_json)
                return json.loads(cleaned_potential_json)
            except (json.JSONDecodeError, re.error):
                pass

    return default_value