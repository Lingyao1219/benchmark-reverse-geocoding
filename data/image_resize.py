import os
import shutil
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

def downsize_images_in_folder(source_folder, destination_folder, size_threshold_mb=1, quality=85):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created directory: {destination_folder}")

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.heic')
    size_threshold_bytes = size_threshold_mb * 1024 * 1024

    print(f"Scanning '{source_folder}' for images larger than {size_threshold_mb}MB...")

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)

        if os.path.isdir(source_path):
            continue

        if filename.lower().endswith(image_extensions):
            try:
                file_size = os.path.getsize(source_path)
                if file_size == 0:
                    print(f"INFO: Skipped '{filename}' because it is an empty file.")
                    continue

                if file_size > size_threshold_bytes:
                    with Image.open(source_path) as img:
                        new_width = img.width // 2
                        new_height = img.height // 2
                        
                        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        if resized_img.mode != 'RGB':
                           resized_img = resized_img.convert("RGB")
                        
                        base, _ = os.path.splitext(filename)
                        destination_path = os.path.join(destination_folder, f"{base}.jpg")
                        
                        resized_img.save(destination_path, 'JPEG', quality=quality)

                        new_file_size_kb = os.path.getsize(destination_path) / 1024
                        print(f"SUCCESS: Resized '{filename}' -> Saved as '{os.path.basename(destination_path)}' ({new_file_size_kb:.2f} KB)")
                else:
                    shutil.copy2(source_path, os.path.join(destination_folder, filename))
                    print(f"INFO: Copied '{filename}' (already small enough).")
            
            except Image.UnidentifiedImageError:
                print(f"ERROR: Could not process '{filename}'. The file is likely corrupted or is an unsupported format.")
            except Exception as e:
                print(f"ERROR: An unexpected error occurred with '{filename}'. Reason: {e}")

if __name__ == '__main__':
    SOURCE_IMAGE_FOLDER = 'dataset3'
    DESTINATION_IMAGE_FOLDER = 'dataset3_new'
    
    if not os.path.isdir(SOURCE_IMAGE_FOLDER):
        print(f"FATAL: Source folder '{SOURCE_IMAGE_FOLDER}' not found. Please check the path.")
    else:
        downsize_images_in_folder(SOURCE_IMAGE_FOLDER, DESTINATION_IMAGE_FOLDER, quality=85)
        print("\nProcessing complete.")