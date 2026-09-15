from __future__ import annotations

import re

import numpy as np

PROMPT_MAX = 600  # chars; whisper keeps ~224 tokens of prompt


def clean_text(text: str, hallucinations: list[str]) -> str:
    out = text
    for h in hallucinations:
        out = re.sub(re.escape(h) + r"[.…!]*", "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def build_prompt(participants: list[str], glossary: list[str]) -> str:
    names = ["Владислав"] + [p for p in participants if p and p != "Владислав"]
    parts = [f"Участники: {', '.join(names)}."]
    if glossary:
        parts.append(f"Термины: {', '.join(glossary)}.")
    prompt = " ".join(parts)
    return prompt if len(prompt) <= PROMPT_MAX else prompt[:PROMPT_MAX].rsplit(",", 1)[0] + "."


class Transcriber:
    def __init__(self, model_name: str, language: str, hallucinations: list[str]):
        import mlx_whisper

        self.mlx_whisper = mlx_whisper
        self.model_name = model_name
        self.language = language
        self.hallucinations = hallucinations

    def __call__(self, audio: np.ndarray, prompt: str = "") -> str:
        result = self.mlx_whisper.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            path_or_hf_repo=self.model_name,
            language=self.language,
            initial_prompt=prompt or None,
            fp16=True,
            condition_on_previous_text=False,
        )
        return clean_text(result.get("text", ""), self.hallucinations)
