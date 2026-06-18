import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# Global state to track active generation progress
generation_state: Dict[str, Any] = {
    "status": "idle",       # idle, running, success, failed
    "stage": "",            # Story Analysis, Image Generation, etc.
    "progress": 0,          # 0 to 100
    "logs": [],
    "total_cost": 0.0,
    "book_pdf": "",
    "error": None
}

app = FastAPI(title="AI Storybook Generator Analytics Dashboard")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared lock for thread-safe operations on state
state_lock = threading.Lock()

def update_status(stage: str, progress: int, log_msg: str = None, total_cost: float = None, book_pdf: str = None, status: str = None):
    """Helper to update global state in a thread-safe manner."""
    with state_lock:
        if status:
            generation_state["status"] = status
        generation_state["stage"] = stage
        generation_state["progress"] = progress
        if log_msg:
            timestamp = time.strftime("%H:%M:%S")
            generation_state["logs"].append(f"[{timestamp}] {log_msg}")
        if total_cost is not None:
            generation_state["total_cost"] = total_cost
        if book_pdf:
            generation_state["book_pdf"] = book_pdf

class GenerateRequest(BaseModel):
    title: str
    author: str
    story_text: str
    mode: str = "ai_enhanced" # classic vs ai_enhanced
    age_group: str = "6-8"
    book_format: str = "a4" # a4, a5, square
    layout_mode: str = "3" # 1, 2, 3 (auto)
    image_fit_mode: str = "contain" # contain, cover
    style_idx: int = 1 # watercolor, folk, modern, dreamy
    placeholders: bool = False
    keep_consistent_look: bool = True
    seed: int = 12345
    gemini_image_model: str = "imagen-4.0-generate-001"

def run_pipeline_thread(req: GenerateRequest, api_key: str):
    """Runs the book generation in a background thread without blocking the FastAPI event loop."""
    try:
        update_status("Story Analysis", 10, "Starting Story Analysis Skill...", status="running")
        
        # Import dynamically here to avoid circular imports
        from book_maker import (
            StoryAnalysisSkill,
            PromptOptimizationSkill,
            CharacterConsistencySkill,
            CostMonitoringSkill,
            build_suggestions,
            ensure_font,
            get_pdf_style,
            slugify,
            render_pdf,
            Automatic1111Client,
            GeminiImagesClient,
            create_placeholder_image
        )
        
        # 1. Setup Context
        context = {
            "story_text": req.story_text,
            "mode": req.mode,
            "max_scenes": 14,
            "content_type": "story",
            "api_key": api_key,
            "output_dir": "output",
            "book_title": req.title,
            "author": req.author,
            "main_character_name": "", # Will be extracted by Story Analysis LLM in AI mode
            "main_character_type": "",
            "main_character_description": "",
            "use_existing_images": False,
            "images_generated_count": 0,
            "use_ip_adapter": req.keep_consistent_look,
            "skills_metrics": [],
            "scenes_count": 0
        }

        # 2. Run Story Analysis Skill
        story_skill = StoryAnalysisSkill()
        context = story_skill.run(context)
        context["skills_metrics"].append(story_skill.get_metrics())

        title = context.get("book_title") or req.title
        scenes = context.get("scenes", [])
        
        update_status("Story Analysis", 30, f"Extracted {len(scenes)} scenes. Title: '{title}'. Style suggestion: '{context.get('suggested_style')}'")

        # 3. Setup Style
        main_name = context.get("main_character_name") or "Mila"
        main_type = context.get("main_character_type") or "girl"
        main_desc = context.get("main_character_description") or "little girl"
        
        # Select predefined style options
        suggestions = build_suggestions(main_name, main_type, main_desc)
        style_idx = req.style_idx - 1
        if style_idx < 0 or style_idx >= len(suggestions):
            style_idx = 0
        chosen_style = suggestions[style_idx]
        context["chosen_style"] = chosen_style

        # 4. Setup Image Generator Clients
        client = Automatic1111Client("http://127.0.0.1:7860", 120000)
        gemini_client = GeminiImagesClient(api_key=api_key, model=req.gemini_image_model)
        
        use_generator = (not req.placeholders) and client.is_available()
        use_gemini_fallback = False
        
        if not use_generator and not req.placeholders:
            if gemini_client.is_available():
                use_gemini_fallback = True
                update_status("API Verification", 40, "Local SD API not available. Using Gemini fallback Imagen API.")
            else:
                req.placeholders = True
                update_status("API Verification", 40, "No SD server and no Gemini API key. Generating placeholder illustrations.")

        context["image_provider"] = (
            "local_stable_diffusion" if use_generator else
            "gemini_fallback" if use_gemini_fallback else
            "placeholder"
        )

        # 5. Run Prompts and Consistency Skills
        update_status("Prompt Optimization", 50, "Optimizing image prompts and consistency guides...")
        prompt_skill = PromptOptimizationSkill()
        context = prompt_skill.run(context)
        context["skills_metrics"].append(prompt_skill.get_metrics())

        consistency_skill = CharacterConsistencySkill()
        context = consistency_skill.run(context)
        context["skills_metrics"].append(consistency_skill.get_metrics())

        # Extract prompt metadata
        scene_prompts = context.get("scene_prompts", {})
        cover_prompt = context.get("cover_prompt", "")
        alwayson_scripts = context.get("alwayson_scripts")

        # 6. Generate Images
        output_dir = Path(context.get("output_dir"))
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup negative prompt
        negative_prompt = "blurry, watermark, logo, signature, text, extra limbs, deformed face, bad anatomy, low quality"
        scene_image_paths = []

        # Generate cover
        update_status("Cover Art Generation", 60, "Generating book cover illustration...")
        cover_path = images_dir / "cover.png"
        
        if req.placeholders:
            create_placeholder_image(cover_path, title, "", 512, 512)
        elif use_generator:
            cover_img = client.txt2img(cover_prompt, negative_prompt, 768, 768, 22, 6.8, req.seed + 999, "Euler a")
            cover_img.save(cover_path)
            context["images_generated_count"] += 1
        elif use_gemini_fallback:
            try:
                cover_img = gemini_client.txt2img(cover_prompt, 768, 768)
                cover_img.save(cover_path)
                context["images_generated_count"] += 1
            except Exception as e:
                update_status("Cover Art Generation", 60, f"Error generating cover image: {e}. Using placeholder.")
                create_placeholder_image(cover_path, title, "", 768, 768)

        # Generate scenes
        total_scenes = len(scenes)
        for idx, scene_text in enumerate(scenes, start=1):
            progress_pct = 60 + int((idx / total_scenes) * 30)
            update_status("Image Generation", progress_pct, f"Generating illustration for page {idx}/{total_scenes}...")
            image_path = images_dir / f"scene_{idx:02d}.png"
            prompt = scene_prompts.get(idx)

            if req.placeholders:
                create_placeholder_image(image_path, f"Scene {idx}", "", 512, 512)
            elif use_generator:
                seed = (req.seed + idx) if req.keep_consistent_look else -1
                image = client.txt2img(prompt, negative_prompt, 512, 768, 22, 6.8, seed, "Euler a", alwayson_scripts)
                image.save(image_path)
                context["images_generated_count"] += 1
            elif use_gemini_fallback:
                try:
                    image = gemini_client.txt2img(prompt, 512, 768)
                    image.save(image_path)
                    context["images_generated_count"] += 1
                except Exception as e:
                    update_status("Image Generation", progress_pct, f"Failed image for scene {idx}: {e}. Falling back to placeholder.")
                    create_placeholder_image(image_path, f"Scene {idx}", "", 512, 768)
            else:
                create_placeholder_image(image_path, f"Scene {idx}", "", 512, 768)

            scene_image_paths.append(image_path)

        # 7. Render PDF
        update_status("PDF Composition", 90, "Assembling illustrations and text into print-ready PDF...")
        font_name = ensure_font(None)
        pdf_style = get_pdf_style(req.age_group, font_name)
        pdf_filename = f"{slugify(title)}_print_ready.pdf"
        pdf_path = output_dir / pdf_filename

        # Layout settings
        include_scene_text = True
        include_full_text_page = req.age_group == "3-5" # full text in A4/A5, false for toddlers usually

        # Determine page size based on format config
        page_size = (595.28, 841.89) # A4 default
        if req.book_format == "a5":
            page_size = (419.53, 595.28)
        elif req.book_format == "square":
            page_size = (595.28, 595.28)

        render_pdf(
            pdf_path=pdf_path,
            title=title,
            author=req.author,
            story_text=req.story_text,
            scenes=scenes,
            scene_images=scene_image_paths,
            cover_image=cover_path,
            pdf_style=pdf_style,
            page_size=page_size,
            include_full_text_page=include_full_text_page,
            include_scene_text=include_scene_text,
            layout_mode=req.layout_mode,
            no_cover_title=False,
            image_fit_mode=req.image_fit_mode,
            content_type="story"
        )

        # 8. Run Cost Monitoring Skill to log results
        update_status("Cost Finalization", 95, "Logging generation metrics and costs...")
        cost_skill = CostMonitoringSkill()
        context["scenes_count"] = len(scenes)
        context = cost_skill.run(context)
        context["skills_metrics"].append(cost_skill.get_metrics())

        run_metrics = context.get("run_metrics", {})
        total_cost = run_metrics.get("metrics", {}).get("total_cost_usd", 0.0)

        update_status(
            "Completed", 100, 
            f"Successfully generated book: {pdf_filename}. Total Cost: ${round(total_cost, 5)}", 
            total_cost=total_cost, 
            book_pdf=pdf_filename, 
            status="success"
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        with state_lock:
            generation_state["status"] = "failed"
            generation_state["error"] = str(e)
            generation_state["logs"].append(f"[ERROR] Generation failed: {e}")

# ==========================================
# FastAPI Route Definitions
# ==========================================

@app.post("/api/generate")
def generate_book(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Spawns a background task to generate the storybook."""
    with state_lock:
        if generation_state["status"] == "running":
            raise HTTPException(status_code=400, detail="A book generation process is already running.")
        
        # Reset state
        generation_state["status"] = "running"
        generation_state["stage"] = "Story Analysis"
        generation_state["progress"] = 0
        generation_state["logs"] = [f"[{time.strftime('%H:%M:%S')}] Received generation request for '{req.title}'."]
        generation_state["total_cost"] = 0.0
        generation_state["book_pdf"] = ""
        generation_state["error"] = None

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY") or ""
    background_tasks.add_task(run_pipeline_thread, req, api_key)
    return {"message": "Book generation started.", "status": "running"}

@app.get("/api/status")
def get_status():
    """Returns the current state and progress of the active generation thread."""
    with state_lock:
        return generation_state

@app.post("/api/cancel")
def cancel_generation():
    """Resets the state back to idle."""
    with state_lock:
        if generation_state["status"] == "running":
            generation_state["status"] = "failed"
            generation_state["logs"].append(f"[{time.strftime('%H:%M:%S')}] Generation cancelled by user.")
            return {"message": "Process cancelled."}
        return {"message": "No active process to cancel."}

@app.get("/api/history")
def get_history():
    """Loads and returns the metrics history from metrics_history.json."""
    history_file = Path("output/metrics_history.json")
    if not history_file.exists():
        return []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

@app.get("/api/config")
def get_config():
    """Returns the current pricing configuration from env variables."""
    # We reload values using os.getenv to show latest changes
    return {
        "gemini_api_key_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")),
        "prices": {
            "input_1m": float(os.getenv("GEMINI_PRICE_INPUT_1M", 0.075)),
            "output_1m": float(os.getenv("GEMINI_PRICE_OUTPUT_1M", 0.30)),
            "cached_1m": float(os.getenv("GEMINI_PRICE_CACHED_1M", 0.01875)),
            "imagen_image": float(os.getenv("IMAGEN_PRICE_PER_IMAGE", 0.03))
        }
    }

# Mount static file endpoints to allow downloading the generated PDFs and images
output_path = Path("output")
output_path.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")
