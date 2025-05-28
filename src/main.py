import os
import base64
import mimetypes
import json
import config
from typing import List, Dict, Any
from utils import query_llm, parse_json
from prompt import system_prompt, image_location_prompt


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
        "error": None
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
        response = query_llm(
            text_prompt=image_location_prompt,
            system_prompt=system_prompt,
            image_base64_data=image_base64,
            image_mime_type=image_type,
            model=config.MODEL,
            provider=config.DEFAULT_PROVIDER,  # Pass the provider explicitly
        )
        result["raw_response"] = response
        
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
        
    image_files = []
    for f_name in os.listdir(config.IMAGE_INPUT_FOLDER):
        if f_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            image_files.append(os.path.join(config.IMAGE_INPUT_FOLDER, f_name))

    if not image_files:
        print(f"No image files found in '{config.IMAGE_INPUT_FOLDER}'.")
        return

    results = []
    print(f"Found {len(image_files)} images to process.")

    for image_path in image_files:
        result = process_image(image_path)
        results.append(result)
        if result.get("location_info"):
            print(f"  -> Guessed Address for {result['image_file']}: {result['location_info'].get('address', 'N/A')}")
        if result.get("error"):
             print(f"  -> Error for {result['image_file']}: {result['error']}")

    # Output as JSONL
    output_path = os.path.join(os.getcwd(), config.RESULTS_OUTPUT_FILE)
    try:
        with open(output_path, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"\nProcessing complete. Results saved to '{output_path}' in JSONL format.")
    except IOError as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    print("Starting image location guessing process...")
    print(f"Reading images from: '{os.path.abspath(config.IMAGE_INPUT_FOLDER)}'")
    main()