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

        resp = requests.post(
            f"{self.base_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("images"):
            raise RuntimeError("Image generation returned no images.")

        raw_b64 = data["images"][0]
        img_bytes = base64.b64decode(raw_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img


class OpenAIImagesClient:
    def __init__(self, api_key: str, timeout: int, model: str) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _size_for_request(width: int, height: int) -> str:
        if width == height:
            return "1024x1024"
        if width > height:
            return "1536x1024"
        return "1024x1536"

    def txt2img(self, prompt: str, width: int, height: int) -> Image.Image:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "size": self._size_for_request(width, height),
            "response_format": "b64_json",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        b64_data = data.get("data", [{}])[0].get("b64_json")
        if not b64_data:
            raise RuntimeError("OpenAI image generation returned no image data.")

        raw = base64.b64decode(b64_data)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return default
    return value


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


def build_suggestions(main_name: str, main_type: str) -> List[CharacterSuggestion]:
    base = f"main character is {main_name}, a {main_type}, friendly, readable silhouette"

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
        "children book illustration, one clear scene, "
        f"{style.prompt_fragment}, "
        "keep exact same main character design as previous pages, "
        f"scene description: {scene_text}, "
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

    c.drawImage(
        img_reader,
        draw_x,
        draw_y,
        width=draw_w,
        height=draw_h,
        preserveAspectRatio=True,
        mask="auto",
    )


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
) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=page_size)
    page_w, page_h = page_size
    margin = 32

    fit_cover_image(c, cover_image, 0, 0, page_w, page_h)
    c.setFillColorRGB(1, 1, 1)
    c.setFont(pdf_style.font_name, pdf_style.title_size)
    c.drawCentredString(page_w / 2, page_h - 110, title)
    c.setFont(pdf_style.font_name, max(14, pdf_style.body_size - 2))
    c.drawCentredString(page_w / 2, page_h - 145, f"by {author}")
    c.showPage()

    if include_full_text_page:
        draw_poem_pages(c, title, author, story_text, pdf_style, page_size)

    total_story_pages = len(scenes)
    for idx, (scene_text, scene_image) in enumerate(zip(scenes, scene_images), start=1):
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
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds.")
    parser.add_argument("--image-width", type=int, default=1792, help="Generated image width in pixels.")
    parser.add_argument("--image-height", type=int, default=2560, help="Generated image height in pixels.")
    parser.add_argument("--steps", type=int, default=35, help="Sampling steps.")
    parser.add_argument("--cfg-scale", type=float, default=7.0, help="CFG scale.")
    parser.add_argument("--sampler", type=str, default="DPM++ 2M Karras", help="Sampler name.")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed for consistent character look.")
    parser.add_argument("--font-path", type=str, default="", help="Optional TTF path for custom text font.")
    parser.add_argument("--placeholders", action="store_true", help="Skip AI generation and use placeholders.")
    parser.add_argument(
        "--fallback-provider",
        type=str,
        default="auto",
        choices=["placeholder", "openai", "auto"],
        help="Fallback provider when local SD API is unavailable.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="",
        help="OpenAI API key for fallback image generation (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--openai-image-model",
        type=str,
        default="gpt-image-1",
        help="OpenAI image model for fallback provider.",
    )
    parser.add_argument(
        "--allow-placeholder-fallback",
        action="store_true",
        help="Allow automatic placeholder fallback if local SD API is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
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

    chosen_format = choose_book_format(args.book_format)

    suggestions = build_suggestions(main_name, main_type)
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

    client = Automatic1111Client(args.sd_base_url, args.timeout)
    openai_client = OpenAIImagesClient(
        api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        timeout=args.timeout,
        model=args.openai_image_model,
    )
    use_generator = (not args.placeholders) and client.is_available()
    use_openai_fallback = False
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

        if args.fallback_provider in {"openai", "auto"}:
            if openai_client.is_available():
                use_openai_fallback = True
                provider_transitions.append(
                    {
                        "from": "local_stable_diffusion",
                        "to": "openai_fallback",
                        "reason": "local_sd_unavailable",
                    }
                )
                print("Using OpenAI Images fallback provider.")
            else:
                if args.fallback_provider == "openai":
                    print("OpenAI fallback provider is selected, but API key is missing.")
                    fallback_answer = ask("Continue with placeholder illustrations? (y/n)", "y").lower()
                    if fallback_answer not in {"y", "yes"}:
                        raise SystemExit(
                            "OpenAI fallback requires OPENAI_API_KEY or --openai-api-key."
                        )
                    provider_transitions.append(
                        {
                            "from": "local_stable_diffusion",
                            "to": "placeholder",
                            "reason": "openai_key_missing_user_confirmed_placeholder",
                        }
                    )
                    print("Using placeholder images because OpenAI API key is not configured.")
                else:
                    provider_transitions.append(
                        {
                            "from": "local_stable_diffusion",
                            "to": "placeholder",
                            "reason": "local_sd_unavailable_openai_key_missing",
                        }
                    )
                    print("OpenAI API key is not configured. Auto mode will continue with placeholders.")

        if not use_openai_fallback:
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

    scene_image_paths: List[Path] = []
    prompt_log = {
        "title": title,
        "author": author,
        "input_file": str(input_path),
        "book_format": chosen_format.__dict__,
        "image_provider": (
            "local_stable_diffusion"
            if use_generator
            else "openai_fallback"
            if use_openai_fallback
            else "placeholder"
        ),
            "provider_transitions": provider_transitions,
        "style": chosen_style.__dict__,
        "scenes": [],
    }

    for index, scene in enumerate(scenes, start=1):
        image_path = images_dir / f"scene_{index:02d}.png"
        prompt = build_scene_prompt(chosen_style, scene)

        if use_generator:
            seed = args.seed + index
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
        elif use_openai_fallback:
            image = openai_client.txt2img(
                prompt=prompt,
                width=args.image_width,
                height=args.image_height,
            )
            image.save(image_path)
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
        cover_img = client.txt2img(
            prompt=cover_prompt,
            negative_prompt=negative_prompt,
            width=args.image_width,
            height=args.image_height,
            steps=max(args.steps, 40),
            cfg_scale=args.cfg_scale,
            seed=args.seed + 999,
            sampler_name=args.sampler,
        )
        cover_img.save(cover_path)
    elif use_openai_fallback:
        cover_img = openai_client.txt2img(
            prompt=cover_prompt,
            width=args.image_width,
            height=args.image_height,
        )
        cover_img.save(cover_path)
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
    )

    prompts_path = output_dir / f"{slugify(title)}_generation_prompts.json"
    prompts_path.write_text(json.dumps(prompt_log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"PDF: {pdf_path}")
    print(f"Images: {images_dir}")
    print(f"Prompt log: {prompts_path}")


if __name__ == "__main__":
    main()
