from .base_skill import BaseSkill

class PromptOptimizationSkill(BaseSkill):
    """
    Skill to construct and optimize image generation prompts.
    Integrates style fragments, character visual descriptions, and scene actions.
    """
    def __init__(self, name: str = "Prompt Optimization") -> None:
        super().__init__(name)

    def execute(self, context: dict) -> dict:
        mode = context.get("mode", "classic")
        scenes = context.get("scenes", [])
        scenes_visuals = context.get("scenes_visuals", {})
        chosen_style = context.get("chosen_style") # CharacterSuggestion object or dict
        
        # Style details
        style_fragment = getattr(chosen_style, "prompt_fragment", "") if hasattr(chosen_style, "prompt_fragment") else context.get("style_prompt_fragment", "")
        style_name = getattr(chosen_style, "name", "") if hasattr(chosen_style, "name") else context.get("style_name", "")

        # 1. Generate Character Design Prompt
        context["character_design_prompt"] = (
            f"Portrait character reference sheet of the main character, "
            "children book illustration style, solid light background, "
            f"{style_fragment}, "
            "centered character design reference, full body visible, high detail, print quality, no text on image"
        )

        # 2. Generate Scene Prompts
        scene_prompts = {}
        for index, scene_text in enumerate(scenes, start=1):
            if mode == "ai_enhanced" and index in scenes_visuals:
                # Use the AI-generated visual description as the primary action
                action = scenes_visuals[index]
            else:
                # Classic mode fallback: use raw scene text
                action = f"Scene action: {scene_text}"

            prompt = (
                f"{action}, children book illustration, one clear scene, "
                f"{style_fragment}, "
                "main character design consistent with previous pages, "
                "high detail, print quality, no text on image"
            )
            scene_prompts[index] = prompt

        context["scene_prompts"] = scene_prompts

        # 3. Generate Cover Prompt
        title = context.get("book_title", "My AI Book")
        story_text = context.get("story_text", "")
        short_story = " ".join(story_text.split())[:320]

        context["cover_prompt"] = (
            "children book COVER illustration, centered composition, "
            f"{style_fragment}, "
            f"inspired by this story: {short_story}, "
            f"book title concept: {title}, "
            "space for title text at top, print quality, no watermark"
        )

        return context
