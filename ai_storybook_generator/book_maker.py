#!/usr/bin/env python3
"""
Local children's book generator for short stories and songs/poems.

Pipeline:
1) Ask for story text.
2) Propose multiple consistent character styles.
3) Extract scene candidates from the text.
4) Generate one illustration per scene (+ cover) via local Stable Diffusion API.
5) Export a print-ready PDF book with embedded font and page layout.

No story content is uploaded by this script. It works with local files and local API endpoints.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4, A5
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from google import genai
from google.genai import types

@dataclass
class CharacterSuggestion:
    name: str
    palette: str
    clothing: str
    visual_style: str
    prompt_fragment: str

@dataclass
class PdfStyle:
    font_name: str
    body_size: int
    title_size: int
    line_gap: int


@dataclass
class BookFormat:
    code: str
    label: str
    page_size: Tuple[float, float]


class Automatic1111Client:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/sdapi/v1/samplers", timeout=self.timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def txt2img(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        sampler_name: str,
    ) -> Image.Image:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "sampler_name": sampler_name,
            "batch_size": 1,
            "n_iter": 1,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=self.timeout,
            )
        except requests.ReadTimeout as exc:
            raise requests.ReadTimeout(
                f"txt2img request timed out after {self.timeout}s. "
                "On CPU, generation can take longer. Increase --timeout (for example 900) "
                "or reduce steps/resolution."
            ) from exc

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            body = resp.text[:1000] if resp is not None else ""
            raise requests.HTTPError(f"{exc}; response body: {body}", response=resp) from exc
        data = resp.json()
        if not data.get("images"):
            raise RuntimeError("Image generation returned no images.")

        raw_b64 = data["images"][0]
        img_bytes = base64.b64decode(raw_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img


class GeminiImagesClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def is_available(self) -> bool:
        return self.client is not None

    def txt2img(self, prompt: str, width: int, height: int) -> Image.Image:
        import time
        # Map width/height to closest aspect ratio supported by Imagen 3/4:
        # "1:1", "3:4", "4:3", "9:16", "16:9"
        aspect_ratio = "1:1"
        if width == height:
            aspect_ratio = "1:1"
        elif width > height:
            if abs((width / height) - (4 / 3)) < abs((width / height) - (16 / 9)):
                aspect_ratio = "4:3"
            else:
                aspect_ratio = "16:9"
        else:
            if abs((height / width) - (4 / 3)) < abs((height / width) - (16 / 9)):
                aspect_ratio = "3:4"
            else:
                aspect_ratio = "9:16"

        max_retries = 3
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                result = self.client.models.generate_images(
                    model=self.model,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type="image/png",
                        aspect_ratio=aspect_ratio,
                    )
                )
                if result.generated_images:
                    generated_image = result.generated_images[0]
                    image = Image.open(io.BytesIO(generated_image.image.image_bytes)).convert("RGB")
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                    return image
                else:
                    print(f"\n  [Warning] Attempt {attempt}/{max_retries}: Image generation returned no images (possible transient error).")
            except Exception as e:
                last_exception = e
                print(f"\n  [Warning] Attempt {attempt}/{max_retries} failed with error: {e}")
            
            if attempt < max_retries:
                time.sleep(2 * attempt)

        if last_exception:
            raise last_exception
        raise RuntimeError("Google GenAI image generation returned no images after retries.")


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    print(f"{prompt}{suffix}: ", end="", flush=True)
    value = input().strip()
    if not value and default is not None:
        return default
    return value


def ask_int(prompt: str, default: int, min_value: int, max_value: int) -> int:
    while True:
        raw = ask(prompt, str(default)).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value < min_value or value > max_value:
            print(f"Please enter a value between {min_value} and {max_value}.")
            continue
        return value


def ask_float(prompt: str, default: float, min_value: float, max_value: float) -> float:
    while True:
        raw = ask(prompt, str(default)).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if value < min_value or value > max_value:
            print(f"Please enter a value between {min_value} and {max_value}.")
            continue
        return value

def ask_generation_settings(args: argparse.Namespace) -> None:
    print("\nImage generation setup:")
    print("1. Recommended setup (no manual tuning)")
    print("2. Advanced setup (fine control)")

    setup_mode = ask("Choose setup mode (1/2)", "1").strip()
    if setup_mode not in {"1", "2"}:
        setup_mode = "1"

    if setup_mode == "1":
        # CPU-safe defaults for local generation (works better on non-CUDA machines).
        args.image_width = 512
        args.image_height = 768
        args.steps = 22
        args.cfg_scale = 6.8
        args.sampler = "Euler a"
        print("\nUsing recommended CPU-safe values:")
        print("- Resolution: 512 x 768")
        print("- Steps: 22")
        print("- CFG scale: 6.8")
        print("- Sampler: Euler a")
    else:
        print("\nAdvanced setup:")
        print("Tip: larger images and more steps give more detail but take longer.")
        print("Pixel guide: 1024x1536 is a good portrait page; 1536x1024 is landscape.")
        print("Bigger numbers mean larger images (and slower generation).")
        args.image_width = ask_int("Image width (pixels, e.g. 1024)", args.image_width, 256, 4096)
        args.image_height = ask_int("Image height (pixels, e.g. 1536)", args.image_height, 256, 4096)
        args.steps = ask_int("Detail level (steps)", args.steps, 1, 150)
        args.cfg_scale = ask_float(
            "Prompt strength (CFG, usually 6-8; higher is NOT always better)",
            args.cfg_scale,
            1.0,
            30.0,
        )
        args.sampler = ask(
            "Sampler (example: DPM++ 2M Karras, Euler a)",
            args.sampler,
        ).strip() or args.sampler

    print("\nCharacter consistency:")
    print("1. Keep the same character look across pages (recommended)")
    print("2. Allow more variation from page to page")
    consistency_mode = ask("Choose consistency mode (1/2)", "1").strip()
    if consistency_mode not in {"1", "2"}:
        consistency_mode = "1"

    args.keep_consistent_look = consistency_mode == "1"
    if args.keep_consistent_look:
        args.seed = ask_int("Base seed", args.seed, 0, 2147483647)

    print("\nUsing generation settings:")
    print(f"- Resolution: {args.image_width} x {args.image_height}")
    print(f"- Steps: {args.steps}")
    print(f"- CFG scale: {args.cfg_scale}")
    print(f"- Sampler: {args.sampler}")
    print(f"- Character consistency: {'on' if args.keep_consistent_look else 'variation allowed'}")
    print(f"- Seed: {args.seed if args.keep_consistent_look else 'random each page'}")


def ask_gemini_fallback_consent() -> bool:
    print("\nPrivacy notice: Gemini fallback sends prompts to Google Gemini API.")
    print(
        "This may include scene text and a short story summary used for the cover illustration prompt."
    )
    answer = ask("Allow Gemini fallback for image generation? (y/n)", "n").strip().lower()
    return answer in {"y", "yes"}


def ask_multiline_story() -> str:
    print("Paste your story/poem text. Finish with a line that only says END.")
    lines: List[str] = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def ask_text_file_path(initial_path: str = "") -> Path:
    while True:
        raw_path = initial_path.strip() if initial_path else ask("Path to .txt file")
        path = Path(raw_path).expanduser()

        if path.suffix.lower() != ".txt":
            print("Please choose a .txt file.")
            initial_path = ""
            continue
        if not path.exists() or not path.is_file():
            print("File does not exist. Try again.")
            initial_path = ""
            continue
        return path


def get_book_format_options() -> List[BookFormat]:
    return [
        BookFormat(code="a4", label="A4 Portrait (210 x 297 mm)", page_size=A4),
        BookFormat(code="a5", label="A5 Portrait (148 x 210 mm)", page_size=A5),
        BookFormat(code="square", label="Square 210 x 210 mm", page_size=(595.28, 595.28)),
    ]


def choose_book_format(default_code: str = "a4") -> BookFormat:
    options = get_book_format_options()
    print("\nBook formats:")
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option.label}")

    default_index = next((i for i, opt in enumerate(options, start=1) if opt.code == default_code), 1)

    while True:
        raw = ask("Choose format (number)", str(default_index))
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(options):
                return options[i - 1]
        print("Enter a valid format number.")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "book"


def extract_scenes(text: str, max_scenes: int) -> List[str]:
    stanzas = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
    if len(stanzas) >= 2:
        scenes = stanzas
    else:
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if len(lines) >= 4:
            chunk_size = 2
            scenes = [" ".join(lines[i : i + chunk_size]) for i in range(0, len(lines), chunk_size)]
        else:
            sentences = [
                x.strip() for x in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if x.strip()
            ]
            scenes = sentences if sentences else [text.strip()]

    deduped: List[str] = []
    seen = set()
    for scene in scenes:
        key = scene.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(scene)

    if len(deduped) > max_scenes:
        return deduped[:max_scenes]
    return deduped


def build_suggestions(main_name: str, main_type: str, main_description: str) -> List[CharacterSuggestion]:
    base = f"main character is {main_name}, a {main_type} (described as: {main_description}), friendly, readable silhouette"

    return [
        CharacterSuggestion(
            name="Warm Watercolor Explorer",
            palette="honey yellow, leaf green, sky blue",
            clothing="striped scarf, soft jacket, little backpack",
            visual_style="watercolor, soft paper texture, expressive eyes",
            prompt_fragment=(
                f"{base}, watercolor children book style, round shapes, "
                "gentle lighting, hand-painted texture"
            ),
        ),
        CharacterSuggestion(
            name="Folk Tale Hero",
            palette="ruby red, deep blue, warm cream",
            clothing="traditional-inspired vest, embroidered details",
            visual_style="storybook gouache, decorative motifs, playful",
            prompt_fragment=(
                f"{base}, folk-inspired children illustration, gouache painting, "
                "ornamental patterns, rich but clean composition"
            ),
        ),
        CharacterSuggestion(
            name="Modern Graphic Adventurer",
            palette="mint, coral, sunflower yellow",
            clothing="hoodie, shorts, colorful sneakers",
            visual_style="flat + textured shapes, bold outlines",
            prompt_fragment=(
                f"{base}, modern graphic children illustration, bold outline, "
                "paper cut texture, vibrant contrast"
            ),
        ),
        CharacterSuggestion(
            name="Dreamy Night Traveler",
            palette="navy, silver, moonlight cyan",
            clothing="starry cloak, comfy boots",
            visual_style="soft glow, cinematic composition, magical",
            prompt_fragment=(
                f"{base}, dreamy illustrated book style, moonlight glow, "
                "soft gradients, magical atmosphere"
            ),
        ),
    ]


def select_suggestion(suggestions: List[CharacterSuggestion]) -> CharacterSuggestion:
    print("\nSuggested illustration styles:")
    for idx, s in enumerate(suggestions, start=1):
        print(f"{idx}. {s.name}")
        print(f"   Palette: {s.palette}")
        print(f"   Clothing: {s.clothing}")
        print(f"   Look: {s.visual_style}")

    while True:
        raw = ask("Choose illustration style (number)", "1")
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(suggestions):
                return suggestions[i - 1]
        print("Enter a valid number.")


def create_placeholder_image(path: Path, title: str, scene_text: str, width: int, height: int) -> None:
    img = Image.new("RGB", (width, height), color=(245, 242, 233))
    draw = ImageDraw.Draw(img)

    # Simple illustration-like fallback scene.
    sky_h = int(height * 0.62)
    draw.rectangle([(0, 0), (width, sky_h)], fill=(188, 222, 255))
    draw.rectangle([(0, sky_h), (width, height)], fill=(152, 210, 132))

    sun_r = max(28, width // 18)
    sun_x = width - sun_r * 2 - 70
    sun_y = 60
    draw.ellipse([(sun_x, sun_y), (sun_x + sun_r * 2, sun_y + sun_r * 2)], fill=(255, 212, 90))

    # Character in center.
    cx = width // 2
    cy = int(height * 0.57)
    head_r = max(42, width // 24)
    body_w = head_r * 2
    body_h = head_r * 3
    draw.ellipse([(cx - head_r, cy - body_h - head_r * 2), (cx + head_r, cy - body_h)], fill=(255, 224, 189), outline=(80, 60, 40), width=3)
    draw.rounded_rectangle(
        [(cx - body_w // 2, cy - body_h), (cx + body_w // 2, cy)],
        radius=18,
        fill=(236, 116, 106),
        outline=(90, 50, 45),
        width=3,
    )
    draw.line([(cx - body_w // 2, cy - body_h // 2), (cx - body_w, cy - body_h // 4)], fill=(90, 50, 45), width=5)
    draw.line([(cx + body_w // 2, cy - body_h // 2), (cx + body_w, cy - body_h // 4)], fill=(90, 50, 45), width=5)
    draw.line([(cx - body_w // 4, cy), (cx - body_w // 3, cy + head_r)], fill=(60, 45, 40), width=6)
    draw.line([(cx + body_w // 4, cy), (cx + body_w // 3, cy + head_r)], fill=(60, 45, 40), width=6)

    try:
        font_big = ImageFont.truetype("arial.ttf", size=max(24, width // 28))
        font_small = ImageFont.truetype("arial.ttf", size=max(18, width // 45))
    except OSError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((60, 60), title, fill=(35, 35, 35), font=font_big)
    wrapped = textwrap.fill(scene_text, width=70)
    # Keep placeholder images text-free to avoid duplicated text in the PDF.
    if scene_text:
        draw.text((60, height - 260), wrapped, fill=(35, 35, 35), font=font_small)
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(100, 90, 70), width=4)

    img.save(path)


def build_scene_prompt(style: CharacterSuggestion, scene_text: str) -> str:
    return (
        f"Scene action: {scene_text}, "
        "children book illustration, one clear scene, "
        f"{style.prompt_fragment}, "
        "main character design consistent with previous pages, "
        "high detail, print quality, no text on image"
    )


def build_cover_prompt(style: CharacterSuggestion, story_text: str, title: str) -> str:
    short_story = " ".join(story_text.split())
    short_story = short_story[:320]
    return (
        "children book COVER illustration, centered composition, "
        f"{style.prompt_fragment}, "
        f"inspired by this story: {short_story}, "
        f"book title concept: {title}, "
        "space for title text at top, print quality, no watermark"
    )


def build_character_design_prompt(style: CharacterSuggestion) -> str:
    return (
        f"Portrait character reference sheet of the main character, "
        "children book illustration style, solid light background, "
        f"{style.prompt_fragment}, "
        "centered character design reference, full body visible, high detail, print quality, no text on image"
    )


def ensure_font(font_path: str | None) -> str:
    candidates = [
        font_path,
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.exists():
            try:
                pdfmetrics.registerFont(TTFont("BookFont", str(p)))
                return "BookFont"
            except Exception:
                continue

    return "Helvetica"


def get_pdf_style(age_group: str, font_name: str) -> PdfStyle:
    if age_group == "3-5":
        return PdfStyle(font_name=font_name, body_size=22, title_size=42, line_gap=10)
    if age_group == "9-12":
        return PdfStyle(font_name=font_name, body_size=16, title_size=34, line_gap=6)
    return PdfStyle(font_name=font_name, body_size=19, title_size=38, line_gap=8)


def fit_cover_image(
    c: canvas.Canvas,
    image_path: Path,
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
) -> None:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        iw, ih = rgb.size
        img_reader = ImageReader(rgb)

    scale = max(box_w / iw, box_h / ih)
    draw_w = iw * scale
    draw_h = ih * scale
    draw_x = box_x + (box_w - draw_w) / 2
    draw_y = box_y + (box_h - draw_h) / 2

    c.saveState()
    path = c.beginPath()
    path.rect(box_x, box_y, box_w, box_h)
    c.clipPath(path, stroke=0, fill=0)

    c.drawImage(
        img_reader,
        draw_x,
        draw_y,
        width=draw_w,
        height=draw_h,
        preserveAspectRatio=True,
        mask="auto",
    )
    c.restoreState()


def wrap_lines(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []

    for word in words:
        trial = " ".join(current + [word])
        trial_width = pdfmetrics.stringWidth(trial, font_name, font_size)
        if trial_width <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def wrap_poem_line(line: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    """Wrap a single poem line while preserving line-by-line verse structure."""
    stripped = line.rstrip()
    if not stripped:
        return [""]

    words = stripped.split()
    if not words:
        return [""]

    wrapped: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width or not current:
            current.append(word)
            continue
        wrapped.append(" ".join(current))
        current = [word]

    if current:
        wrapped.append(" ".join(current))

    return wrapped


def draw_poem_pages(
    c: canvas.Canvas,
    title: str,
    author: str,
    story_text: str,
    pdf_style: PdfStyle,
    page_size: Tuple[float, float],
) -> None:
    """Render full poem text across one or more pages, preserving stanza breaks."""
    page_w, page_h = page_size
    margin_x = 50
    top_y = page_h - 88
    bottom_y = 58
    line_height = pdf_style.body_size + pdf_style.line_gap
    stanza_gap = max(line_height // 2, 8)
    max_width = page_w - 2 * margin_x

    lines = story_text.splitlines()
    if not lines:
        lines = [story_text]

    def start_page(page_number: int) -> float:
        c.setFillColorRGB(1, 1, 1)
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont(pdf_style.font_name, pdf_style.title_size - 10)
        c.drawString(margin_x, page_h - 56, title)
        c.setFont(pdf_style.font_name, max(12, pdf_style.body_size - 3))
        c.drawRightString(page_w - margin_x, page_h - 56, f"Author: {author} | Text {page_number}")
        c.setFont(pdf_style.font_name, pdf_style.body_size)
        return top_y

    page_number = 1
    y = start_page(page_number)

    for original_line in lines:
        wrapped_line_parts = wrap_poem_line(
            original_line,
            pdf_style.font_name,
            pdf_style.body_size,
            max_width,
        )

        for part in wrapped_line_parts:
            needed_gap = stanza_gap if part == "" else line_height
            if y - needed_gap < bottom_y:
                c.showPage()
                page_number += 1
                y = start_page(page_number)

            if part == "":
                y -= stanza_gap
            else:
                c.drawString(margin_x, y, part)
                y -= line_height

    c.setFont(pdf_style.font_name, 10)
    c.drawString(margin_x, 34, "Full song/poem text is shown without shortening.")
    c.showPage()


def render_pdf(
    pdf_path: Path,
    title: str,
    author: str,
    story_text: str,
    scenes: List[str],
    scene_images: List[Path],
    cover_image: Path,
    pdf_style: PdfStyle,
    page_size: Tuple[float, float],
    include_full_text_page: bool,
    include_scene_text: bool,
    layout_mode: str = "1",
    no_cover_title: bool = False,
    cover_author_font: str | None = None,
    cover_author_y: float = 145.0,
    cover_author_size: int | None = None,
) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=page_size)
    page_w, page_h = page_size
    margin = 32

    fit_cover_image(c, cover_image, 0, 0, page_w, page_h)
    c.setFillColorRGB(1, 1, 1)
    
    if not no_cover_title:
        c.setFont(pdf_style.font_name, pdf_style.title_size)
        c.drawCentredString(page_w / 2, page_h - 110, title)
        
    author_font = pdf_style.font_name
    author_size = cover_author_size if cover_author_size is not None else max(14, pdf_style.body_size - 2)
    
    if cover_author_font:
        try:
            if os.path.exists(cover_author_font):
                font_key = "CustomCoverAuthorFont"
                pdfmetrics.registerFont(TTFont(font_key, cover_author_font))
                author_font = font_key
            else:
                author_font = cover_author_font
        except Exception as e:
            print(f"Warning: Could not register custom cover author font {cover_author_font}: {e}")

    c.setFont(author_font, author_size)
    c.drawCentredString(page_w / 2, page_h - cover_author_y, f"by {author}")
    c.showPage()

    if include_full_text_page:
        draw_poem_pages(c, title, author, story_text, pdf_style, page_size)

    total_story_pages = len(scenes)
    for idx, (scene_text, scene_image) in enumerate(zip(scenes, scene_images), start=1):
        current_layout = layout_mode
        if current_layout == "3":
            lines = wrap_lines(scene_text, pdf_style.font_name, pdf_style.body_size, page_w - 2 * margin)
            current_layout = "2" if len(lines) > 5 else "1"

        if current_layout == "2":
            # Text Page
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
            if include_scene_text:
                c.setFillColorRGB(0.12, 0.12, 0.12)
                c.setFont(pdf_style.font_name, pdf_style.body_size)
                lines = wrap_lines(scene_text, pdf_style.font_name, pdf_style.body_size, page_w - 2 * margin)
                total_text_h = len(lines) * (pdf_style.body_size + pdf_style.line_gap)
                text_y = (page_h + total_text_h) / 2
                for line in lines:
                    c.drawCentredString(page_w / 2, text_y, line)
                    text_y -= (pdf_style.body_size + pdf_style.line_gap)
            c.showPage()
            
            # Illustration Page
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
            fit_cover_image(c, scene_image, 0, 0, page_w, page_h)
            c.setFillColorRGB(0, 0, 0)
            c.setFont(pdf_style.font_name, 11)
            c.drawRightString(page_w - margin, 22, f"{idx}/{total_story_pages}")
            c.showPage()
            
        else:
            # Layout 1: Image top, text bottom
            image_box_x = margin
            image_box_y = page_h * 0.34
            image_box_w = page_w - 2 * margin
            image_box_h = page_h * 0.62

            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
            fit_cover_image(c, scene_image, image_box_x, image_box_y, image_box_w, image_box_h)

            if include_scene_text:
                c.setFillColorRGB(0.12, 0.12, 0.12)
                c.setFont(pdf_style.font_name, pdf_style.body_size)
                lines = wrap_lines(scene_text, pdf_style.font_name, pdf_style.body_size, page_w - 2 * margin)
                text_y = page_h * 0.28
                max_lines = 6
                for line in lines[:max_lines]:
                    c.drawString(margin, text_y, line)
                    text_y -= (pdf_style.body_size + pdf_style.line_gap)

            c.setFont(pdf_style.font_name, 11)
            c.drawRightString(page_w - margin, 22, f"{idx}/{total_story_pages}")
            c.showPage()

    c.save()

 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local print-ready children book PDF.")
    parser.add_argument("--input-file", type=str, default="", help="Path to UTF-8 story text file.")
    parser.add_argument(
        "--book-format",
        type=str,
        default="a4",
        choices=["a4", "a5", "square"],
        help="Book page format.",
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for assets and PDF.")
    parser.add_argument("--max-scenes", type=int, default=14, help="Maximum number of story scenes/pages.")
    parser.add_argument("--sd-base-url", type=str, default="http://127.0.0.1:7860", help="Local SD API URL.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=120000,
        help="HTTP timeout in seconds (CPU generation can be slow; 120000 is recommended).",
    )
    parser.add_argument("--image-width", type=int, default=1792, help="Generated image width in pixels.")
    parser.add_argument("--image-height", type=int, default=2560, help="Generated image height in pixels.")
    parser.add_argument("--steps", type=int, default=35, help="Sampling steps.")
    parser.add_argument("--cfg-scale", type=float, default=7.0, help="CFG scale.")
    parser.add_argument("--sampler", type=str, default="DPM++ 2M Karras", help="Sampler name.")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed for consistent character look.")
    parser.add_argument("--font-path", type=str, default="", help="Optional TTF path for custom text font.")
    parser.add_argument(
        "--no-cover-title",
        action="store_true",
        help="Skip drawing the book title on the cover page.",
    )
    parser.add_argument(
        "--cover-author-font",
        type=str,
        default="",
        help="Optional TTF path or name for the cover author font.",
    )
    parser.add_argument(
        "--cover-author-y",
        type=float,
        default=145.0,
        help="Vertical distance from the top of the cover page to draw the author name.",
    )
    parser.add_argument(
        "--cover-author-size",
        type=int,
        default=None,
        help="Optional font size for the cover author name.",
    )
    parser.add_argument("--placeholders", action="store_true", help="Skip AI generation and use placeholders.")
    parser.add_argument(
        "--fallback-provider",
        type=str,
        default="auto",
        choices=["placeholder", "gemini", "auto"],
        help="Fallback provider when local SD API is unavailable.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default="",
        help="Gemini API key for fallback image generation (or set GEMINI_API_KEY / GENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--gemini-image-model",
        type=str,
        default="imagen-4.0-generate-001",
        help="Gemini image model for fallback provider.",
    )
    parser.add_argument(
        "--allow-placeholder-fallback",
        action="store_true",
        help="Allow automatic placeholder fallback if local SD API is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    # Ensure stdout is line-buffered/flushed immediately in non-interactive terminals or IDE runners
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, io.UnsupportedOperation):
        pass

    args = parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    input_path = ask_text_file_path(args.input_file)
    story_text = input_path.read_text(encoding="utf-8").strip()

    if not story_text:
        raise SystemExit("No story text provided.")

    title = ask("Book title", "My AI Book")
    author = ask("Author", "Unknown")
    age_group = ask("Age group (3-5 / 6-8 / 9-12)", "6-8")
    main_name = ask("Main character name", "Mila")
    main_type = ask("Main character type (girl, boy, animal, creature)", "girl")
    main_description = ask("Detailed character description", "little silver dragon with shiny round sapphire-blue eyes, two tiny gold horns, a red velvet backpack, and small translucent wings")
    content_type = ask("Is this a story or a song/poem? (story/song)", "story").strip().lower()
    if content_type not in {"story", "song"}:
        print("Invalid choice. Using 'story'.")
        content_type = "story"

    song_illustration_mode = "multiple"
    if content_type == "song":
        song_illustration_mode = ask(
            "For a song/poem, create one illustration for the whole text or multiple per scene? (one/multiple)",
            "one",
        ).strip().lower()
        if song_illustration_mode not in {"one", "multiple"}:
            print("Invalid choice. Using 'one'.")
            song_illustration_mode = "one"

    layout_mode = "1"
    if content_type == "story":
        print("\nPage layout:")
        print("1. Text below illustration (best for short sentences)")
        print("2. Text on left page, full illustration on right page (best for long text)")
        print("3. Auto (decide based on text length)")
        layout_mode = ask("Choose layout (1/2/3)", "3").strip()
        if layout_mode not in {"1", "2", "3"}:
            layout_mode = "3"

    chosen_format = choose_book_format(args.book_format)

    suggestions = build_suggestions(main_name, main_type, main_description)
    chosen_style = select_suggestion(suggestions)

    scenes = extract_scenes(story_text, args.max_scenes)
    if not scenes:
        raise SystemExit("Could not extract any scenes from text.")

    if content_type == "song" and song_illustration_mode == "one":
        scenes = [" ".join(story_text.split())]

    print("\nDetected scenes:")
    for i, s in enumerate(scenes, start=1):
        short = s if len(s) <= 120 else s[:117] + "..."
        print(f"{i}. {short}")

    proceed = ask("Continue with these scenes? (y/n)", "y").lower()
    if proceed not in {"y", "yes"}:
        raise SystemExit("Canceled by user.")

    ask_generation_settings(args)

    client = Automatic1111Client(args.sd_base_url, args.timeout)
    gemini_client = GeminiImagesClient(
        api_key=args.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY") or "",
        model=args.gemini_image_model,
    )
    use_generator = (not args.placeholders) and client.is_available()
    use_gemini_fallback = False
    provider_transitions: List[dict] = []

    if args.placeholders:
        provider_transitions.append(
            {
                "from": "local_stable_diffusion",
                "to": "placeholder",
                "reason": "manual_placeholders_flag",
            }
        )

    if not use_generator and not args.placeholders:
        print(
            "\nLocal Stable Diffusion API is unavailable "
            f"({args.sd_base_url})."
        )

        if args.fallback_provider in {"gemini", "auto"}:
            if gemini_client.is_available():
                if ask_gemini_fallback_consent():
                    use_gemini_fallback = True
                    provider_transitions.append(
                        {
                            "from": "local_stable_diffusion",
                            "to": "gemini_fallback",
                            "reason": "local_sd_unavailable_user_consented_gemini",
                        }
                    )
                    print("Using Gemini Images fallback provider.")
                else:
                    print("Gemini fallback not enabled because consent was not granted.")
            else:
                if args.fallback_provider == "gemini":
                    print("Gemini fallback provider is selected, but API key is missing.")
                    fallback_answer = ask("Continue with placeholder illustrations? (y/n)", "y").lower()
                    if fallback_answer not in {"y", "yes"}:
                        raise SystemExit(
                            "Gemini fallback requires GEMINI_API_KEY, GENAI_API_KEY or --gemini-api-key."
                        )
                    provider_transitions.append(
                        {
                            "from": "local_stable_diffusion",
                            "to": "placeholder",
                            "reason": "gemini_key_missing_user_confirmed_placeholder",
                        }
                    )
                    print("Using placeholder images because Gemini API key is not configured.")
                else:
                    provider_transitions.append(
                        {
                            "from": "local_stable_diffusion",
                            "to": "placeholder",
                            "reason": "local_sd_unavailable_gemini_key_missing",
                        }
                    )
                    print("Gemini API key is not configured. Auto mode will continue with placeholders.")

        if not use_gemini_fallback:
            fallback_reason = "--allow-placeholder-fallback is enabled"
            if not args.allow_placeholder_fallback:
                fallback_answer = ask("Continue with placeholder illustrations? (y/n)", "y").lower()
                if fallback_answer not in {"y", "yes"}:
                    raise SystemExit(
                        "Local Stable Diffusion API is unavailable. "
                        "Start your local server and retry, or use --placeholders / --allow-placeholder-fallback."
                    )
                fallback_reason = "interactive fallback is confirmed"
            if not provider_transitions:
                provider_transitions.append(
                    {
                        "from": "local_stable_diffusion",
                        "to": "placeholder",
                        "reason": (
                            "local_sd_unavailable_allow_placeholder_flag"
                            if args.allow_placeholder_fallback
                            else "local_sd_unavailable_user_confirmed_placeholder"
                        ),
                    }
                )
            print(
                "Local Stable Diffusion API was not found at "
                f"{args.sd_base_url}. Using placeholder images because {fallback_reason}."
            )

    negative_prompt = (
        "blurry, watermark, logo, signature, text, extra limbs, deformed face, "
        "bad anatomy, low quality"
    )

    # Character Design Approval Phase
    if use_generator or use_gemini_fallback:
        print("\n=== Character Design Approval Phase ===")
        while True:
            preview_path = images_dir / "character_design_preview.png"
            preview_prompt = build_character_design_prompt(chosen_style)
            
            print(f"\nGenerating character design preview with style: '{chosen_style.name}'")
            print(f"Character description: '{main_description}'")
            
            if use_generator:
                print(f"Generating preview with seed {args.seed}...")
                try:
                    preview_img = client.txt2img(
                        prompt=preview_prompt,
                        negative_prompt=negative_prompt,
                        width=args.image_width,
                        height=args.image_height,
                        steps=args.steps,
                        cfg_scale=args.cfg_scale,
                        seed=args.seed,
                        sampler_name=args.sampler,
                    )
                    preview_img.save(preview_path)
                    print(f"Character design preview saved to {preview_path}")
                except Exception as e:
                    print(f"Error generating character preview via Stable Diffusion: {e}")
            elif use_gemini_fallback:
                print("Generating preview with Gemini fallback...")
                try:
                    preview_img = gemini_client.txt2img(
                        prompt=preview_prompt,
                        width=args.image_width,
                        height=args.image_height,
                    )
                    preview_img.save(preview_path)
                    print(f"Character design preview saved to {preview_path}")
                except Exception as e:
                    print(f"Error generating character preview via Gemini: {e}")
            
            satisfied = ask("\nAre you satisfied with this character design? (y/n)", "y").strip().lower()
            if satisfied in {"y", "yes"}:
                print("Character design approved!")
                break
                
            print("\nHow would you like to adjust the design?")
            print("1. Enter character description tweaks (appends to character description)")
            print("2. Change character seed (currently: {})".format(args.seed))
            print("3. Change character style (re-choose illustration style)")
            print("4. Edit entire character description from scratch")
            print("5. Keep current design and proceed anyway")
            
            choice = ask("Choose an option (1/2/3/4/5)", "5").strip()
            if choice == "1":
                tweak = ask("Enter description tweak (e.g. 'wearing a blue hat', 'red scales')", "").strip()
                if tweak:
                    main_description = f"{main_description}, {tweak}"
                    suggestions = build_suggestions(main_name, main_type, main_description)
                    for s in suggestions:
                        if s.name == chosen_style.name:
                            chosen_style = s
                            break
            elif choice == "2":
                args.seed = ask_int("Enter new seed/offset (-1 for random)", args.seed + 1, -1, 2147483647)
            elif choice == "3":
                suggestions = build_suggestions(main_name, main_type, main_description)
                chosen_style = select_suggestion(suggestions)
            elif choice == "4":
                new_desc = ask("Enter new character description", main_description).strip()
                if new_desc:
                    main_description = new_desc
                    suggestions = build_suggestions(main_name, main_type, main_description)
                    for s in suggestions:
                        if s.name == chosen_style.name:
                            chosen_style = s
                            break
            elif choice == "5":
                break

    scene_image_paths: List[Path] = []
    prompt_log = {
        "title": title,
        "author": author,
        "input_file": str(input_path),
        "book_format": chosen_format.__dict__,
        "image_provider": (
            "local_stable_diffusion"
            if use_generator
            else "gemini_fallback"
            if use_gemini_fallback
            else "placeholder"
        ),
        "provider_transitions": provider_transitions,
        "generation_settings": {
            "image_width": args.image_width,
            "image_height": args.image_height,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "sampler": args.sampler,
            "seed": args.seed,
            "keep_consistent_look": getattr(args, "keep_consistent_look", True),
        },
        "style": chosen_style.__dict__,
        "scenes": [],
    }

    for index, scene in enumerate(scenes, start=1):
        image_path = images_dir / f"scene_{index:02d}.png"
        prompt = build_scene_prompt(chosen_style, scene)

        if use_generator:
            keep_consistent_look = getattr(args, "keep_consistent_look", True)
            seed = (args.seed + index) if keep_consistent_look else -1
            image = client.txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=args.image_width,
                height=args.image_height,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=seed,
                sampler_name=args.sampler,
            )
            image.save(image_path)
        elif use_gemini_fallback:
            try:
                image = gemini_client.txt2img(
                    prompt=prompt,
                    width=args.image_width,
                    height=args.image_height,
                )
                image.save(image_path)
            except Exception as e:
                print(f"\n[WARNING] Failed to generate image for Scene {index} due to error: {e}")
                print("Falling back to placeholder image for this scene to continue book generation.")
                create_placeholder_image(
                    path=image_path,
                    title=f"Scene {index}",
                    scene_text="",
                    width=args.image_width,
                    height=args.image_height,
                )
        else:
            create_placeholder_image(
                path=image_path,
                title=f"Scene {index}",
                scene_text="",
                width=args.image_width,
                height=args.image_height,
            )

        scene_image_paths.append(image_path)
        prompt_log["scenes"].append({"index": index, "scene_text": scene, "prompt": prompt})
        print(f"Generated: {image_path}")

    cover_prompt = build_cover_prompt(chosen_style, story_text, title)
    cover_path = images_dir / "cover.png"

    if use_generator:
        keep_consistent_look = getattr(args, "keep_consistent_look", True)
        cover_seed = (args.seed + 999) if keep_consistent_look else -1
        cover_img = client.txt2img(
            prompt=cover_prompt,
            negative_prompt=negative_prompt,
            width=args.image_width,
            height=args.image_height,
            steps=max(args.steps, 40),
            cfg_scale=args.cfg_scale,
            seed=cover_seed,
            sampler_name=args.sampler,
        )
        cover_img.save(cover_path)
    elif use_gemini_fallback:
        try:
            cover_img = gemini_client.txt2img(
                prompt=cover_prompt,
                width=args.image_width,
                height=args.image_height,
            )
            cover_img.save(cover_path)
        except Exception as e:
            print(f"\n[WARNING] Failed to generate cover image due to error: {e}")
            print("Falling back to placeholder image for the cover.")
            create_placeholder_image(
                path=cover_path,
                title=title,
                scene_text="",
                width=args.image_width,
                height=args.image_height,
            )
    else:
        create_placeholder_image(
            path=cover_path,
            title=title,
            scene_text="",
            width=args.image_width,
            height=args.image_height,
        )

    prompt_log["cover_prompt"] = cover_prompt

    font_name = ensure_font(args.font_path or None)
    pdf_style = get_pdf_style(age_group.strip(), font_name)

    pdf_filename = f"{slugify(title)}_print_ready.pdf"
    pdf_path = output_dir / pdf_filename

    render_pdf(
        pdf_path=pdf_path,
        title=title,
        author=author,
        story_text=story_text,
        scenes=scenes,
        scene_images=scene_image_paths,
        cover_image=cover_path,
        pdf_style=pdf_style,
        page_size=chosen_format.page_size,
        include_full_text_page=(content_type == "song"),
        include_scene_text=(content_type == "story"),
        layout_mode=layout_mode,
        no_cover_title=args.no_cover_title,
        cover_author_font=args.cover_author_font or None,
        cover_author_y=args.cover_author_y,
        cover_author_size=args.cover_author_size,
    )

    prompts_path = output_dir / f"{slugify(title)}_generation_prompts.json"
    prompts_path.write_text(json.dumps(prompt_log, ensure_ascii=False, indent=2), encoding="utf-8")

    while True:
        satisfied = ask("\nAre you satisfied with the book? (y/n)", "y").strip().lower()
        if satisfied in {"y", "yes"}:
            break

        print("\nWhat would you like to do?")
        print("1. Re-generate specific illustration(s)")
        print("2. Modify PDF layout / styling / cover properties")
        print("3. Done (Exit and save changes)")

        choice = ask("Choose an option (1/2/3)", "3").strip()
        if choice == "1":
            print("\nPage list:")
            print("  0. Cover page")
            for idx, scene in enumerate(scenes, start=1):
                short = scene if len(scene) <= 80 else scene[:77] + "..."
                print(f"  {idx}. Scene {idx}: {short}")

            selection = ask("Enter the numbers of illustrations to re-generate (separated by commas, e.g. 0, 2)", "").strip()
            if not selection:
                continue

            indices_to_regen = []
            try:
                for x in selection.split(","):
                    val = int(x.strip())
                    if 0 <= val <= len(scenes):
                        indices_to_regen.append(val)
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
                continue

            if not indices_to_regen:
                print("No valid page numbers selected.")
                continue

            for idx in indices_to_regen:
                if idx == 0:
                    print(f"\n--- Re-generating Cover Image ---")
                    print(f"Original prompt: {cover_prompt}")
                    tweak = ask("Enter additional instructions to append (or press Enter to keep prompt)", "").strip()
                    active_prompt = cover_prompt
                    if tweak:
                        active_prompt = f"{cover_prompt}, {tweak}"

                    if use_generator:
                        keep_consistent_look = getattr(args, "keep_consistent_look", True)
                        cover_seed = ask_int("Enter new seed/offset (-1 for random)", args.seed + 999 + 1, -1, 2147483647)
                        print(f"Generating with seed {cover_seed}...")
                        cover_img = client.txt2img(
                            prompt=active_prompt,
                            negative_prompt=negative_prompt,
                            width=args.image_width,
                            height=args.image_height,
                            steps=max(args.steps, 40),
                            cfg_scale=args.cfg_scale,
                            seed=cover_seed,
                            sampler_name=args.sampler,
                        )
                        cover_img.save(cover_path)
                    elif use_gemini_fallback:
                        try:
                            print("Generating with Gemini fallback...")
                            cover_img = gemini_client.txt2img(
                                prompt=active_prompt,
                                width=args.image_width,
                                height=args.image_height,
                            )
                            cover_img.save(cover_path)
                        except Exception as e:
                            print(f"Failed to generate cover image: {e}")
                    else:
                        print("No image provider available.")
                    
                    prompt_log["cover_prompt"] = active_prompt
                else:
                    scene_idx = idx - 1
                    scene_text = scenes[scene_idx]
                    scene_image_path = scene_image_paths[scene_idx]
                    orig_prompt = build_scene_prompt(chosen_style, scene_text)
                    print(f"\n--- Re-generating Scene {idx} ---")
                    print(f"Original prompt: {orig_prompt}")
                    tweak = ask("Enter additional instructions to append (or press Enter to keep prompt)", "").strip()
                    active_prompt = orig_prompt
                    if tweak:
                        active_prompt = f"{orig_prompt}, {tweak}"

                    if use_generator:
                        keep_consistent_look = getattr(args, "keep_consistent_look", True)
                        scene_seed = ask_int("Enter new seed/offset (-1 for random)", args.seed + idx + 1, -1, 2147483647)
                        print(f"Generating with seed {scene_seed}...")
                        image = client.txt2img(
                            prompt=active_prompt,
                            negative_prompt=negative_prompt,
                            width=args.image_width,
                            height=args.image_height,
                            steps=args.steps,
                            cfg_scale=args.cfg_scale,
                            seed=scene_seed,
                            sampler_name=args.sampler,
                        )
                        image.save(scene_image_path)
                    elif use_gemini_fallback:
                        try:
                            print("Generating with Gemini fallback...")
                            image = gemini_client.txt2img(
                                prompt=active_prompt,
                                width=args.image_width,
                                height=args.image_height,
                            )
                            image.save(scene_image_path)
                        except Exception as e:
                            print(f"Failed to generate image for Scene {idx}: {e}")
                    else:
                        print("No image provider available.")
                    
                    prompt_log["scenes"][scene_idx]["prompt"] = active_prompt

            # Re-render PDF with the updated images
            render_pdf(
                pdf_path=pdf_path,
                title=title,
                author=author,
                story_text=story_text,
                scenes=scenes,
                scene_images=scene_image_paths,
                cover_image=cover_path,
                pdf_style=pdf_style,
                page_size=chosen_format.page_size,
                include_full_text_page=(content_type == "song"),
                include_scene_text=(content_type == "story"),
                layout_mode=layout_mode,
                no_cover_title=args.no_cover_title,
                cover_author_font=args.cover_author_font or None,
                cover_author_y=args.cover_author_y,
                cover_author_size=args.cover_author_size,
            )
            prompts_path.write_text(json.dumps(prompt_log, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nPDF and prompts log updated successfully!")

        elif choice == "2":
            print("\nModify PDF layout / styling settings:")
            current_no_title = args.no_cover_title
            no_title_ans = ask(f"Skip cover title? (y/n, currently {'y' if current_no_title else 'n'})", "y" if current_no_title else "n").strip().lower()
            args.no_cover_title = no_title_ans in {"y", "yes"}

            current_author_font = args.cover_author_font or ""
            author_font_ans = ask(f"Cover author font path (currently '{current_author_font}', press Enter to keep)", current_author_font).strip()
            args.cover_author_font = author_font_ans

            current_author_y = args.cover_author_y
            args.cover_author_y = ask_float(f"Cover author Y position (currently {current_author_y})", current_author_y, 0.0, 1500.0)

            current_author_size = args.cover_author_size
            current_size_str = str(current_author_size) if current_author_size is not None else "default"
            author_size_ans = ask(f"Cover author font size (currently {current_size_str}, Enter to keep)", current_size_str).strip()
            if author_size_ans.lower() != "default" and author_size_ans.isdigit():
                args.cover_author_size = int(author_size_ans)
            elif author_size_ans.lower() == "default":
                args.cover_author_size = None

            current_font_path = args.font_path or ""
            font_path_ans = ask(f"Main body font path (currently '{current_font_path}', press Enter to keep)", current_font_path).strip()
            args.font_path = font_path_ans

            font_name = ensure_font(args.font_path or None)
            pdf_style = get_pdf_style(age_group.strip(), font_name)

            render_pdf(
                pdf_path=pdf_path,
                title=title,
                author=author,
                story_text=story_text,
                scenes=scenes,
                scene_images=scene_image_paths,
                cover_image=cover_path,
                pdf_style=pdf_style,
                page_size=chosen_format.page_size,
                include_full_text_page=(content_type == "song"),
                include_scene_text=(content_type == "story"),
                layout_mode=layout_mode,
                no_cover_title=args.no_cover_title,
                cover_author_font=args.cover_author_font or None,
                cover_author_y=args.cover_author_y,
                cover_author_size=args.cover_author_size,
            )
            print(f"\nPDF updated with styling changes!")

        elif choice == "3":
            break

    print("\nDone.")
    print(f"PDF: {pdf_path}")
    print(f"Images: {images_dir}")
    print(f"Prompt log: {prompts_path}")


if __name__ == "__main__":
    main()
