import json
import re
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from .base_skill import BaseSkill

class SceneModel(BaseModel):
    index: int = Field(description="Sequential page index starting from 1")
    text: str = Field(description="The exact text or stanza to be displayed on this page of the book")
    visual_description: str = Field(description="Detailed visual prompt description of the action occurring in this scene, suitable for image generation")

class StoryAnalysisModel(BaseModel):
    suggested_title: str = Field(description="A creative title for the story book")
    suggested_style: str = Field(description="Suggested illustration style matching the book mood (e.g., watercolor, folk art, modern graphic, dreamy)")
    character_name: str = Field(description="Name of the main character")
    character_type: str = Field(description="Type of the main character (e.g. girl, boy, animal, dragon)")
    character_description: str = Field(description="Detailed physical description of the character (clothing, colors, specific details) to maintain visual consistency")
    scenes: List[SceneModel] = Field(description="List of chronological scenes representing pages of the book")

class StoryAnalysisSkill(BaseSkill):
    """
    Skill to split a story text into scenes and analyze characters.
    Supports Classic Regex-based parsing and AI-Enhanced Gemini-based analysis.
    """
    def __init__(self, name: str = "Story Analysis") -> None:
        super().__init__(name)

    def execute(self, context: dict) -> dict:
        story_text = context.get("story_text", "").strip()
        mode = context.get("mode", "classic")
        max_scenes = context.get("max_scenes", 14)
        content_type = context.get("content_type", "story")

        if not story_text:
            raise ValueError("Story text cannot be empty in StoryAnalysisSkill.")

        if mode == "ai_enhanced" and context.get("api_key"):
            try:
                self._run_ai_analysis(context, story_text, max_scenes, content_type)
                return context
            except Exception as e:
                print(f"\n[Warning] AI Story Analysis failed: {e}. Falling back to Classic Regex parsing.")
                # Fallback to classic mode
                context["mode_fallback_triggered"] = True

        # Run Classic Analysis (Regex)
        self._run_classic_analysis(context, story_text, max_scenes, content_type)
        return context

    def _run_ai_analysis(self, context: dict, story_text: str, max_scenes: int, content_type: str) -> None:
        """Helper to invoke Gemini API for story parsing."""
        api_key = context.get("api_key")
        client = genai.Client(api_key=api_key)

        prompt = f"""
        Analyze the following children's {content_type} and structure it into a print-ready illustrated book.
        
        Story Text:
        {story_text}

        Requirements:
        1. Segment the text into a maximum of {max_scenes} logical, chronological scenes. Each scene will represent one book page.
        2. Identify the main character's name, type, and create a very detailed visual description to be used for image prompt consistency (e.g. clothing colors, specific features).
        3. Suggest a visual illustration style that fits the mood of the story.
        4. For each scene, write a visual_description explaining what is happening in the scene for an image generator (like Stable Diffusion). Keep descriptions detailed, clear, and without text elements.
        """

        # Call Gemini 2.5 Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StoryAnalysisModel,
                temperature=0.2,
            )
        )

        # Parse response
        data = json.loads(response.text)

        # Update metrics
        if response.usage_metadata:
            self.metrics["input_tokens"] = response.usage_metadata.prompt_token_count
            self.metrics["output_tokens"] = response.usage_metadata.candidates_token_count
            self.metrics["cached_tokens"] = getattr(response.usage_metadata, "cached_content_token_count", 0)

        # Save to context
        context["book_title"] = context.get("book_title") or data.get("suggested_title", "My AI Book")
        context["suggested_style"] = data.get("suggested_style", "watercolor")
        context["main_character_name"] = context.get("main_character_name") or data.get("character_name", "Mila")
        context["main_character_type"] = context.get("main_character_type") or data.get("character_type", "girl")
        context["main_character_description"] = context.get("main_character_description") or data.get("character_description", "")
        
        # Scenes extraction
        context["scenes"] = [scene.text for scene in data.get("scenes", [])]
        context["scenes_visuals"] = {
            i: scene.visual_description for i, scene in enumerate(data.get("scenes", []), start=1)
        }
        context["scenes_count"] = len(context["scenes"])

    def _run_classic_analysis(self, context: dict, story_text: str, max_scenes: int, content_type: str) -> None:
        """Regex-based scene parsing fallback (corresponds to legacy extract_scenes)."""
        if content_type == "song":
            stanzas = [x.strip() for x in re.split(r"\n\s*\n", story_text) if x.strip()]
            if len(stanzas) >= 2:
                scenes = stanzas
            else:
                lines = [x.strip() for x in story_text.splitlines() if x.strip()]
                scenes = []
                for i in range(0, len(lines), 4):
                    chunk = lines[i : i + 4]
                    scenes.append("\n".join(chunk))
        else:
            stanzas = [x.strip() for x in re.split(r"\n\s*\n", story_text) if x.strip()]
            if len(stanzas) >= 2:
                scenes = stanzas
            else:
                lines = [x.strip() for x in story_text.splitlines() if x.strip()]
                if len(lines) >= 4:
                    chunk_size = 2
                    scenes = [" ".join(lines[i : i + chunk_size]) for i in range(0, len(lines), chunk_size)]
                else:
                    sentences = [
                        x.strip() for x in re.split(r"(?<=[.!?])\s+", story_text.replace("\n", " ")) if x.strip()
                    ]
                    scenes = sentences if sentences else [story_text.strip()]

        deduped: List[str] = []
        seen = set()
        for scene in scenes:
            key = scene.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(scene)

        if len(deduped) > max_scenes:
            deduped = deduped[:max_scenes]

        context["scenes"] = deduped
        context["scenes_count"] = len(deduped)
        # In classic mode, there are no dynamic visual descriptions, so other skills will rely on static rules.
        context["scenes_visuals"] = {}
