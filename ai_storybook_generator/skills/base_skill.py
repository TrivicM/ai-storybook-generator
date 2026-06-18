from abc import ABC, abstractmethod
import time

class BaseSkill(ABC):
    """
    Abstract base class for all agent skills in the pipeline.
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self.execution_time = 0.0
        self.metrics = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0.0,
            "success": True,
            "error": None
        }

    @abstractmethod
    def execute(self, context: dict) -> dict:
        """
        Executes the skill action with the given shared pipeline context.
        Must return the updated context dict.
        """
        pass

    def run(self, context: dict) -> dict:
        """
        Wrapper around execute to measure execution time and handle exceptions safely.
        """
        start_time = time.time()
        self.metrics["success"] = True
        self.metrics["error"] = None
        try:
            context = self.execute(context)
        except Exception as e:
            self.metrics["success"] = False
            self.metrics["error"] = str(e)
            raise e
        finally:
            self.execution_time = time.time() - start_time
        return context

    def get_metrics(self) -> dict:
        """
        Returns execution metrics of the skill.
        """
        return {
            "name": self.name,
            "execution_time_seconds": round(self.execution_time, 3),
            **self.metrics
        }
