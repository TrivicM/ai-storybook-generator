import base64
import os
from pathlib import Path
from .base_skill import BaseSkill

class CharacterConsistencySkill(BaseSkill):
    """
    Skill to manage visual consistency of the main character.
    Reads the approved character preview image, encodes it to base64,
    and sets up the payload configuration for ControlNet IP-Adapter (if using Stable Diffusion).
    """
    def __init__(self, name: str = "Character Consistency") -> None:
        super().__init__(name)

    def execute(self, context: dict) -> dict:
        use_existing = context.get("use_existing_images", False)
        if use_existing:
            return context

        # Default settings for IP-Adapter
        ip_adapter_enabled = context.get("use_ip_adapter", True)
        ip_adapter_weight = float(os.getenv("IP_ADAPTER_WEIGHT", 0.7))
        ip_adapter_guidance_end = float(os.getenv("IP_ADAPTER_GUIDANCE_END", 0.8))

        # Check if reference image exists
        images_dir = Path(context.get("output_dir", "output")) / "images"
        ref_image_path = images_dir / "character_design_preview.png"

        # If IP-Adapter is enabled, and the character sheet exists, configure ControlNet
        if ip_adapter_enabled and ref_image_path.exists() and context.get("image_provider") == "local_stable_diffusion":
            try:
                base64_image = self._image_to_base64(ref_image_path)
                
                # Setup the ControlNet dictionary matching test_ip_adapter.py logic
                alwayson_scripts = {
                    "controlnet": {
                        "args": [
                            {
                                "enabled": True,
                                "module": "ip-adapter_clip_sd15",
                                "model": "ip-adapter-plus_sd15",
                                "weight": ip_adapter_weight,
                                "image": base64_image,
                                "resize_mode": "Crop and Resize",
                                "control_mode": "Balanced",
                                "pixel_perfect": True,
                                "guidance_start": 0.0,
                                "guidance_end": ip_adapter_guidance_end
                            }
                        ]
                    }
                }
                
                context["alwayson_scripts"] = alwayson_scripts
                context["ip_adapter_active"] = True
                print(f"\n[Character Consistency] ControlNet IP-Adapter configured successfully using template: {ref_image_path}")
            except Exception as e:
                print(f"\n[Warning] Failed to configure ControlNet IP-Adapter: {e}. Proceeding without IP-Adapter.")
                context["ip_adapter_active"] = False
        else:
            context["ip_adapter_active"] = False
            if ip_adapter_enabled and context.get("image_provider") == "gemini_fallback":
                print("\n[Character Consistency] Gemini fallback does not support ControlNet. Visual consistency is managed via prompt engineering.")

        return context

    def _image_to_base64(self, image_path: Path) -> str:
        """Helper to convert image to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
