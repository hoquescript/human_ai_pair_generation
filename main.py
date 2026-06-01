import os
import sys
import importlib.metadata

# torchcodec is not in CC's wheelhouse; patch the metadata probe that
# transformers/audio_utils.py runs at import time so it doesn't raise.
_orig_version = importlib.metadata.version


def _patched_version(name: str) -> str:
    if name == "torchcodec":
        return "0.0.0"
    return _orig_version(name)


importlib.metadata.version = _patched_version

import pandas as pd  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def clean_code_output(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def get_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


def get_prompt(code: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a code generator. You are given a piece of code written by a human. Your task is to produce an alternative version of this code. You must output only raw code. Never include markdown, backticks, or explanations.",
        },
        {
            "role": "user",
            "content": f"""You are given a piece of code written by a human. Your task is to produce an alternative version of this code.
            Guidelines:
            - Use the same programming language.
            - Preserve the original functionality and behavior exactly.
            - You may refactor structure, rename variables, adjust formatting, or use equivalent constructs.
            - Do not intentionally simplify or over-complicate the code.
            - Do not add comments or stylistic markers that reveal authorship.

            STRICT OUTPUT FORMAT:
            - Output ONLY raw code.
            - Do NOT include markdown formatting.
            - Do NOT include triple backticks (```).
            - Do NOT include language labels like "java", "python", etc.
            - Do NOT include any explanation or extra text.
            - The output must start directly with code and end with code.

            Original code:
            {code}
        """,
        },
    ]


def generate_ai_pair(tokenizer, model, messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=65536,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
    )
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return clean_code_output(response)


if __name__ == "__main__":
    environment = os.environ.get("ENVIRONMENT", "dev")
    language = os.environ.get("LANGUAGE", "java")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    chunk_index = int(os.environ.get("CHUNK_INDEX", 0))
    total_chunks = int(os.environ.get("TOTAL_CHUNKS", 1))

    df = pd.read_csv(f"data/aidev/{language}.csv")
    sample_df = df.sample(n=5)
    dataframe = sample_df if environment == "dev" else df
    dataframe = dataframe[dataframe["code"].str.len() < 20000]
    human_df = dataframe[dataframe["label"] == 1].copy()

    # Split into chunks if needed
    if total_chunks > 1:
        chunk_size = len(human_df) // total_chunks
        start = chunk_index * chunk_size
        end = start + chunk_size if chunk_index < total_chunks - 1 else len(human_df)
        human_df = human_df.iloc[start:end]
        print(
            f"Chunk {chunk_index + 1}/{total_chunks}: processing rows {start} to {end}"
        )

    # Output path
    output_dir = "data/contrastive-aidev"
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_chunk{chunk_index}" if total_chunks > 1 else ""
    output_path = f"{output_dir}/{language}{suffix}_paired.jsonl"

    # Checkpointing — skip already processed rows
    processed_indices = set()
    if os.path.exists(output_path):
        try:
            existing = pd.read_json(output_path, orient="records", lines=True)
            if "original_index" in existing.columns:
                processed_indices = set(existing["original_index"].tolist())
                print(
                    f"Resuming: {len(processed_indices)} samples already done, skipping."
                )
        except Exception as e:
            print(f"Warning: could not read existing output: {e}. Starting fresh.")

    remaining = human_df[~human_df.index.isin(processed_indices)]
    total = len(human_df)
    remaining_count = len(remaining)

    print(
        f"Generating contrastive pairs for {language}... in mode {environment} by model {model_name}"
    )
    print(
        f"Total: {total} | Done: {total - remaining_count} | Remaining: {remaining_count}"
    )

    if remaining_count == 0:
        print("All samples already processed. Exiting.")
        sys.exit(0)

    tokenizer, model = get_model(model_name)

    for i, (index, row) in enumerate(remaining.iterrows()):
        code_human = row["code"]
        prompt = get_prompt(code_human)
        code_ai = generate_ai_pair(tokenizer, model, prompt)

        # Write each row immediately
        row_out = row.to_dict()
        row_out["contrast"] = code_ai
        row_out["original_index"] = index

        pd.DataFrame([row_out]).to_json(
            output_path,
            orient="records",
            lines=True,
            mode="a",
        )

        print(f"[{i + 1}/{remaining_count}] Sample {index} done.")
