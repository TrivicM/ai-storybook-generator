#!/usr/bin/env python3
import base64
import io
import os
import sys
import requests
from PIL import Image

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def main():
    # 1. Configuration
    sd_url = "http://127.0.0.1:7860"
    
    # Try to find a default image or ask the user
    default_image = "output/images/character_design_preview.png"
    if os.path.exists(default_image):
        ref_image_path = default_image
    else:
        ref_image_path = input("Enter path to a template/character image file: ").strip().strip('"')
        
    if not os.path.exists(ref_image_path):
        print(f"Error: Reference image not found at '{ref_image_path}'")
        sys.exit(1)
        
    prompt = input("Enter prompt for the new scene (e.g. 'a little dragon reading a book under a tree'): ").strip()
    if not prompt:
        prompt = "a little dragon reading a book under a tree, children book illustration, colorful"

    output_path = "ip_adapter_result.png"

    print(f"\nUsing reference image: {ref_image_path}")
    print(f"Using prompt: '{prompt}'")
    
    # Convert image to base64
    base64_image = image_to_base64(ref_image_path)
    
    # 2. Build the payload
    # Note: 'module' is the preprocessor. For SD 1.5, we use 'ip-adapter_clip_sd15'
    # Ask for weight and guidance settings
    print("\nIP-Adapter Settings:")
    weight = float(input("Enter IP-Adapter weight (0.0 to 1.0, recommended 0.7): ") or "0.7")
    guidance_end = float(input("Enter guidance end step (0.0 to 1.0, recommended 0.8): ") or "0.8")
    
    # 2. Build the payload
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, deformed, extra limbs, bad anatomy",
        "steps": 12,
        "cfg_scale": 7.0,
        "width": 512,
        "height": 512,
        "sampler_name": "Euler a",
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "module": "ip-adapter_clip_sd15",
                        "model": "ip-adapter-plus_sd15",
                        "weight": weight,
                        "image": base64_image,
                        "resize_mode": "Crop and Resize",
                        "control_mode": "Balanced",
                        "pixel_perfect": True,
                        "guidance_start": 0.0,
                        "guidance_end": guidance_end
                    }
                ]
            }
        }
    }
    
    # 3. Call the API
    try:
        print(f"Sending request to local Stable Diffusion WebUI at {sd_url}...")
        response = requests.post(f"{sd_url}/sdapi/v1/txt2img", json=payload, timeout=1200)
        response.raise_for_status()
        
        # 4. Save result
        r_json = response.json()
        if "images" in r_json and r_json["images"]:
            image_data = base64.b64decode(r_json["images"][0])
            image = Image.open(io.BytesIO(image_data))
            image.save(output_path)
            print(f"\nSuccess! Generated image saved to: {os.path.abspath(output_path)}")
        else:
            print("\nError: API returned no images. Check WebUI console for errors.")
            
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to Stable Diffusion WebUI at {sd_url}.")
        print("Please verify that your WebUI is running and has --api in its command line arguments.")
    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()
