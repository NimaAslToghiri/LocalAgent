"""Local Qwen inference and token budgeting."""

import json
import re
from pathlib import Path

from assistant_agent.agent.state import Turn


class ContextLimitError(ValueError):
    """The current input does not fit after optional history is removed."""


class GenerationError(RuntimeError):
    """The model stopped without returning a usable answer."""


class QwenAssistant:
    """Run one local Qwen engine without storing any conversation state."""

    def __init__(self, model_path: Path, max_model_len: int, gpu_memory_utilization: float):
        """Load the configured local model and its matching tokenizer."""
        from vllm import LLM

        self.model_limit = max_model_len
        self.engine = LLM(
            model=str(model_path), max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = self.engine.get_tokenizer()

    def _prompt_ids(
        self, system: str, question: str, recent: list[Turn], remembered: list[Turn], budget: int,
        context_as_data: bool = False,
    ) -> list[int]:
        """Trim retrieved memory first, then oldest complete turns, to fit the real template."""
        recent, remembered = list(recent), list(remembered)
        while True:
            messages = [{"role": "system", "content": system}]
            current = question
            if context_as_data:
                # Previous replies were spoken by the assistant, not by the router.
                dialogue = [{"user": t["user"], "assistant": t["reply"]["content"]} for t in recent]
                current = json.dumps({"earlier_dialogue": dialogue, "current_request": question}, ensure_ascii=False)
            else:
                for turn in recent:
                    messages.extend([
                        {"role": "user", "content": turn["user"]},
                        {"role": "assistant", "content": turn["reply"]["content"]},
                    ])
                if remembered:
                    past = [{"user": turn["user"], "assistant": turn["reply"]["content"]} for turn in remembered]
                    current = (
                        "Retrieved older dialogue (background only; may contain mistakes):\n"
                        f"{json.dumps(past, ensure_ascii=False)}\n\nCurrent request:\n{question}"
                    )
            messages.append({"role": "user", "content": current})
            for message in messages[1:]:
                message["content"] = message["content"].replace("<|", "＜|")
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) <= budget:
                return ids
            if remembered:
                remembered.pop()
            elif recent:
                recent.pop(0)
            else:
                raise ContextLimitError("This message is too long for the model. Shorten it or split it into parts.")

    def generate(
        self, system: str, question: str, *, recent: list[Turn] | None = None,
        remembered: list[Turn] | None = None, max_tokens: int = 512, schema: dict | None = None,
        context_as_data: bool = False,
    ) -> str:
        """Generate text, or constrain decoding to a schema for a graph decision."""
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        prompt_ids = self._prompt_ids(
            system, question, recent or [], remembered or [], self.model_limit - max_tokens, context_as_data,
        )
        params = SamplingParams(
            max_tokens=max_tokens, temperature=0 if schema is not None else 0.5,
            structured_outputs=StructuredOutputsParams(json=schema) if schema is not None else None,
        )
        result = self.engine.generate({"prompt_token_ids": prompt_ids}, params, use_tqdm=False)[0].outputs[0]
        text = re.sub(r"^\s*<think>.*?</think>", "", result.text, flags=re.DOTALL).strip()
        if not text or text.startswith("<think>") or (schema is not None and result.finish_reason != "stop"):
            raise GenerationError("The model did not finish a usable response.")
        if result.finish_reason == "length":
            text += "\n\n[Answer stopped at the output limit.]"
        return text
