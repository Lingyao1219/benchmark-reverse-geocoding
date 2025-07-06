import os
import sys
import base64
import mimetypes
import json
import config
from typing import List, Dict, Any
from model import query_llm, parse_json
from prompt import SYSTEM_PROMPT, IMAGE_LOCATION_PROMPT


def get_image_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type:
        return mime_type
    ext = os.path.splitext(image_path)[1].lower()
    common_mime_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"
    }
    return common_mime_types.get(ext)

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def process_image(image_path: str) -> Dict[str, Any]:
    result = {
        "image_file": os.path.basename(image_path),
        "provider": config.DEFAULT_PROVIDER,
        "model_used": config.MODEL,
        "raw_response": None,
        "location_info": None,
        "error": None,
        "usage_info": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        }
    }

    print(f"\nProcessing image: {result['image_file']}")

    image_base64 = encode_image(image_path)
    if not image_base64:
        result["error"] = f"Failed to load image: {image_path}"
        return result

    image_type = get_image_type(image_path)
    if not image_type:
        result["error"] = f"Could not determine image type for: {image_path}"
        return result
    
    try:
        response, usage_info = query_llm(
            text_prompt=IMAGE_LOCATION_PROMPT,
            system_prompt=SYSTEM_PROMPT,
            image_base64_data=image_base64,
            image_mime_type=image_type,
            model=config.MODEL,
            provider=config.DEFAULT_PROVIDER,
        )
        
        result["raw_response"] = response
        result["usage_info"] = usage_info
        
        print(f"  -> Cost: ${usage_info['cost_usd']:.6f}")
        print(f"  -> Tokens: {usage_info['total_tokens']} ({usage_info['input_tokens']} in, {usage_info['output_tokens']} out)")
        
        if response:
            parsed_info = parse_json(response, default_value={"address": "Parse Failed", "reasoning": "Could not parse output."})
            result["location_info"] = parsed_info
        else:
            result["error"] = "Empty response."
            result["location_info"] = {"address": "No Response", "reasoning": "No content returned."}

    except Exception as e:
        print(f"Error during LLM query for {result['image_file']}: {e}")
        result["error"] = str(e)
        result["location_info"] = {"address": "Error", "reasoning": str(e)}
    
    return result

def main():
    if not os.path.exists(config.IMAGE_INPUT_FOLDER):
        print(f"Error: Input folder '{config.IMAGE_INPUT_FOLDER}' not found. Please create it and add images.")
        return
        
    all_image_files = []
    for f_name in os.listdir(config.IMAGE_INPUT_FOLDER):
        if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            all_image_files.append(os.path.join(config.IMAGE_INPUT_FOLDER, f_name))

    if not all_image_files:
        print(f"No image files found in '{config.IMAGE_INPUT_FOLDER}'.")
        return

    output_path = os.path.join(os.getcwd(), config.RESULTS_OUTPUT_FILE)
    processed_images = set()
    if os.path.exists(output_path):
        print(f"Checking for previously processed images in '{output_path}'...")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        if 'image_file' in res:
                            processed_images.add(res['image_file'])
                    except json.JSONDecodeError:
                        continue
        except IOError as e:
            print(f"Warning: Could not read existing results file: {e}")

    image_files_to_process = [
        path for path in all_image_files if os.path.basename(path) not in processed_images
    ]
    
    if not image_files_to_process:
        print("\nAll images in the folder have already been processed.")
        print(f"Results are in '{output_path}'.")
        return

    print(f"\nFound {len(all_image_files)} total images. {len(processed_images)} already processed. {len(image_files_to_process)} to process now.")
    
    total_cost = 0.0
    total_tokens = 0
    successful_processes = 0
    
    print(f"Using model: {config.MODEL} (Provider: {config.DEFAULT_PROVIDER})")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to '{output_path}'")

    for image_path in image_files_to_process:
        result = process_image(image_path)
        
        try:
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
        except IOError as e:
            print(f"  -> CRITICAL: Error saving result to file for {result['image_file']}: {e}")

        if not result.get("error"):
            successful_processes += 1
            total_cost += result["usage_info"]["cost_usd"]
            total_tokens += result["usage_info"]["total_tokens"]
        
        if result.get("location_info"):
            location_info = result['location_info']
            address_parts = []
            if 'reverse_geocoding' in location_info and 'address' in location_info['reverse_geocoding']:
                addr = location_info['reverse_geocoding']['address']
                formatted_address = 'Address not determined'
                
                if isinstance(addr, dict):
                    if addr.get('street'): address_parts.append(addr['street'])
                    if addr.get('city'): address_parts.append(addr['city'])
                    if addr.get('state'): address_parts.append(addr['state'])
                    if addr.get('country'): address_parts.append(addr['country'])
                    if address_parts: formatted_address = ', '.join(address_parts)
                elif isinstance(addr, str):
                    formatted_address = addr
                    
                confidence = location_info['reverse_geocoding'].get('confidence', 'unknown')
                print(f"  -> Guessed Address for {result['image_file']}: {formatted_address} (Confidence: {confidence})")
            else:
                print(f"  -> Guessed Address for {result['image_file']}: Unable to parse location data")
        if result.get("error"):
             print(f"  -> Error for {result['image_file']}: {result['error']}")

    print(f"\n" + "="*50)
    print(f"PROCESSING SUMMARY")
    print(f"="*50)
    print(f"Total images in folder: {len(all_image_files)}")
    print(f"Images processed this run: {len(image_files_to_process)}")
    print(f"Successful processes this run: {successful_processes}")
    print(f"Failed processes this run: {len(image_files_to_process) - successful_processes}")
    print(f"Total cost this run: ${total_cost:.6f}")
    print(f"Total tokens used this run: {total_tokens:,}")
    if successful_processes > 0:
        print(f"Average cost per image: ${total_cost/successful_processes:.6f}")
        print(f"Average tokens per image: {total_tokens//successful_processes:,}")

    print(f"\nProcessing complete. All results have been saved to '{output_path}'.")

def analyze_costs(results_file: str = None):
    """Analyze costs from previous runs"""
    if not results_file:
        results_file = config.RESULTS_OUTPUT_FILE
    
    if not os.path.exists(results_file):
        print(f"Results file '{results_file}' not found.")
        return
    
    total_cost = 0.0
    total_tokens = 0
    results_count = 0
    model_stats = {}
    
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    
                    if 'usage_info' not in result or result.get('error'):
                        continue
                    
                    usage = result['usage_info']
                    model = result.get('model_used', 'unknown')
                    
                    total_cost += usage.get('cost_usd', 0)
                    total_tokens += usage.get('total_tokens', 0)
                    results_count += 1
                    
                    if model not in model_stats:
                        model_stats[model] = {'count': 0, 'total_cost': 0.0, 'total_tokens': 0}
                    
                    model_stats[model]['count'] += 1
                    model_stats[model]['total_cost'] += usage.get('cost_usd', 0)
                    model_stats[model]['total_tokens'] += usage.get('total_tokens', 0)
        
        print(f"\n" + "="*50)
        print(f"COST ANALYSIS FROM {results_file}")
        print(f"="*50)
        print(f"Total successful runs: {results_count}")
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Total tokens: {total_tokens:,}")
        
        if results_count > 0:
            print(f"Average cost per run: ${total_cost/results_count:.6f}")
            print(f"Average tokens per run: {total_tokens//results_count:,}")
        
        print(f"\nBy Model:")
        for model, stats in model_stats.items():
            avg_cost = stats['total_cost'] / stats['count'] if stats['count'] > 0 else 0
            avg_tokens = stats['total_tokens'] // stats['count'] if stats['count'] > 0 else 0
            print(f"  {model}:")
            print(f"    Runs: {stats['count']}")
            print(f"    Total cost: ${stats['total_cost']:.6f}")
            print(f"    Avg cost/run: ${avg_cost:.6f}")
            print(f"    Avg tokens/run: {avg_tokens:,}")
    
    except json.JSONDecodeError as e:
        print(f"Error parsing results file: {e}")
    except Exception as e:
        print(f"Error analyzing costs: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_costs()
    else:
        print("Starting image location guessing process...")
        print(f"Reading images from: '{os.path.abspath(config.IMAGE_INPUT_FOLDER)}'")
        main()