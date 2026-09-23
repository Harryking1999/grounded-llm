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


class TransformersCaller:
    """Local model caller for nodes without a serving process."""

    def __init__(self, model_path: str, *, enable_thinking: bool, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("TransformersCaller requires a CUDA device")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype="auto"
        ).to("cuda").eval()
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> dict:
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=self.enable_thinking,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, input_ids.shape[1]:]
        stop_ids = self.model.generation_config.eos_token_id
        if isinstance(stop_ids, int):
            stop_ids = [stop_ids]
        stopped = bool(len(generated) and int(generated[-1]) in (stop_ids or []))
        response_ids = generated[:-1] if stopped else generated
        return {"text": self.tokenizer.decode(response_ids, skip_special_tokens=False),
                "finish_reason": "stop" if stopped else "length",
                "prompt_tokens": int(input_ids.shape[1]),
                "completion_tokens": int(len(generated))}
