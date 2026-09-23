"""Model caller and strict action parsing; no environment or map logic."""

import json
import urllib.request


def parse_action_id(text: str) -> int:
    # Qwen thinking output may include a closed reasoning section.
    answer = text.split("</think>", 1)[-1].strip()
    value = json.loads(answer)
    if not isinstance(value, dict) or type(value.get("action_id")) is not int:
        raise ValueError("Expected JSON with integer action_id")
    return value["action_id"]


class SGLangCaller:
    def __init__(self, model_path: str, endpoint: str, *, enable_thinking: bool,
                 max_new_tokens: int, timeout_seconds: int = 120):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.endpoint = endpoint.rstrip("/")
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds

    def __call__(self, prompt: str) -> dict:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.enable_thinking,
        )
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        request = urllib.request.Request(
            self.endpoint + "/generate",
            data=json.dumps({"input_ids": ids, "sampling_params": {
                "temperature": 0, "max_new_tokens": self.max_new_tokens,
            }}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.load(response)
        meta = result["meta_info"]
        return {"text": result["text"],
                "finish_reason": meta["finish_reason"]["type"],
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens")}
