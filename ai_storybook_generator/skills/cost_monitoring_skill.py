import json
import os
import time
from pathlib import Path
from .base_skill import BaseSkill

class CostMonitoringSkill(BaseSkill):
    """
    Skill responsible for accumulating metrics from the pipeline,
    calculating financial cost, and saving results to history logs.
    """
    # Default prices (Gemini 2.5 Flash & Imagen)
    # Can be overridden using environment variables in a .env file
    PRICE_LLM_INPUT_1M = float(os.getenv("GEMINI_PRICE_INPUT_1M", 0.075))
    PRICE_LLM_OUTPUT_1M = float(os.getenv("GEMINI_PRICE_OUTPUT_1M", 0.30))
    PRICE_LLM_CACHED_1M = float(os.getenv("GEMINI_PRICE_CACHED_1M", 0.01875))
    PRICE_IMAGEN_PER_IMAGE = float(os.getenv("IMAGEN_PRICE_PER_IMAGE", 0.03))

    def __init__(self, name: str = "Cost Monitoring", history_file: str = "output/metrics_history.json") -> None:
        super().__init__(name)
        self.history_file = Path(history_file)

    def execute(self, context: dict) -> dict:
        """
        Gathers metrics from all executed skills, computes total cost,
        and saves the log to the history file.
        """
        # Read existing pipeline metrics
        skills_metrics = context.get("skills_metrics", [])
        
        # Calculate totals for this run
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        total_images_generated = context.get("images_generated_count", 0)
        image_provider = context.get("image_provider", "placeholder")

        # Sum token metrics from other skills
        for metric in skills_metrics:
            total_input_tokens += metric.get("input_tokens", 0)
            total_output_tokens += metric.get("output_tokens", 0)
            total_cached_tokens += metric.get("cached_tokens", 0)

        # Calculate costs
        llm_input_cost = (total_input_tokens / 1_000_000.0) * self.PRICE_LLM_INPUT_1M
        llm_output_cost = (total_output_tokens / 1_000_000.0) * self.PRICE_LLM_OUTPUT_1M
        llm_cached_cost = (total_cached_tokens / 1_000_000.0) * self.PRICE_LLM_CACHED_1M
        
        # Images are free if using local Stable Diffusion
        image_cost = 0.0
        if image_provider == "gemini_fallback":
            image_cost = total_images_generated * self.PRICE_IMAGEN_PER_IMAGE

        total_cost = llm_input_cost + llm_output_cost + llm_cached_cost + image_cost

        # Record metrics in this skill's metrics
        self.metrics["input_tokens"] = total_input_tokens
        self.metrics["output_tokens"] = total_output_tokens
        self.metrics["cached_tokens"] = total_cached_tokens
        self.metrics["cost_usd"] = total_cost

        # Context details for this book generation run
        run_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "book_title": context.get("book_title", "Unknown"),
            "author": context.get("author", "Unknown"),
            "mode": context.get("mode", "classic"),
            "image_provider": image_provider,
            "scenes_count": context.get("scenes_count", 0),
            "metrics": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "cached_tokens": total_cached_tokens,
                "images_generated": total_images_generated,
                "llm_cost_usd": round(llm_input_cost + llm_output_cost + llm_cached_cost, 6),
                "image_cost_usd": round(image_cost, 4),
                "total_cost_usd": round(total_cost, 6)
            }
        }

        # Save record to metrics history JSON file
        self._write_to_history(run_record)

        # Put the final calculations back in the context
        context["run_metrics"] = run_record
        return context

    def _write_to_history(self, record: dict) -> None:
        """Helper to append log records to the JSON history log file."""
        history = []
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            except Exception:
                history = []

        history.append(record)
        
        # Ensure parent folder exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to write metrics history to file: {e}")
