import json
import re
import ast
from typing import Optional, Dict, Any, List
from openai import OpenAI
import anthropic
import config


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
) -> str:
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
    return response.choices[0].message.content


def query_claude(
    system_prompt: str,
    user_content_blocks: List[Dict[str, Any]],
    model: str,
    temperature: float
) -> str:
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
    for item in response.content:
        if item.type == 'text':
            return item.text
    return ""

def query_togetherai(
    messages_payload: List[Dict[str, Any]],
    model: str,
    temperature: float
) -> str:
    api_key = load_api_key(provider='togetherai')
    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    response = client.chat.completions.create(
        model=model,
        messages=messages_payload,
        temperature=temperature,
    )
    return response.choices[0].message.content

def query_llm(
    text_prompt: str,
    system_prompt: Optional[str] = None,
    image_base64_data: Optional[str] = None,
    image_mime_type: Optional[str] = None,
    model: Optional[str] = config.MODEL,
    provider: Optional[str] = None,
) -> str:
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
        return query_openai(messages_payload, model)

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
        return query_claude(common_system_prompt, user_content_blocks_list, model, config.TEMPERATURE)

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
        return query_togetherai(messages_payload_tg, model, config.TEMPERATURE)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

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

    cleaned_text = text_to_parse
    cleaned_text = cleaned_text.replace("None", "null").replace("True", "true").replace("False", "false")
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
        start_index = -1
        first_brace = text_to_parse.find('{')
        first_bracket = text_to_parse.find('[')

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_index = first_brace
            end_index = text_to_parse.rfind('}')
        elif first_bracket != -1:
            start_index = first_bracket
            end_index = text_to_parse.rfind(']')
        else:
            end_index = -1

        if start_index != -1 and end_index > start_index:
            potential_json = text_to_parse[start_index : end_index+1]
            try:
                cleaned_potential_json = potential_json.replace("None", "null").replace("True", "true").replace("False", "false")
                cleaned_potential_json = re.sub(r'(?<=[{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', cleaned_potential_json)
                if "'" in cleaned_potential_json:
                    cleaned_potential_json = cleaned_potential_json.replace("'", '"')
                cleaned_potential_json = re.sub(r',\s*([}\]])', r'\1', cleaned_potential_json)
                return json.loads(cleaned_potential_json)
            except json.JSONDecodeError:
                pass

    return default_value