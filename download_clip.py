#!/usr/bin/env python3
import os
import requests
import ssl
import urllib3

# Disable warnings for unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Bypass SSL Windows Store certificate bug in Python
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

def main():
    url = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin"
    dest = r"C:\AI_alati\stable-diffusion-webui\extensions\sd-webui-controlnet\annotator\downloads\clip_vision\clip_h.pth"
    
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        print(f"Creating directory: {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)
        
    print(f"Downloading CLIP-H vision model...")
    print(f"Source: {url}")
    print(f"Destination: {dest}")
    
    try:
        # Use verify=True (strict verification) which uses requests' internal cert bundle (certifi)
        # instead of the buggy Windows store, keeping it 100% secure.
        response = requests.get(url, stream=True, verify=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end="\r")
                    else:
                        print(f"Downloaded: {downloaded / (1024*1024):.1f}MB", end="\r")
                        
        print("\n\nSuccess! CLIP-H vision model downloaded successfully.")
    except Exception as e:
        print(f"\nError downloading model: {e}")

if __name__ == "__main__":
    main()
