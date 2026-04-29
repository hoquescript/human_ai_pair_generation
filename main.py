import pandas as pd
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.messages import SystemMessage, HumanMessage

import os


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
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs=dict(
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=2048,
            return_full_text=False,
        ),
    )
    model = ChatHuggingFace(llm=llm, stop=["```"])
    return model


def get_prompt(code: str) -> list[dict[str, str]]:
    return [
        SystemMessage(
            "You are a code generator. You are given a piece of code written by a human. Your task is to produce an alternative version of this code. You must output only raw code. Never include markdown, backticks, or explanations. "
        ),
        HumanMessage(f"""You are given a piece of code written by a human. Your task is to produce an alternative version of this code.
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
        """),
    ]


def generate_ai_pair(model: ChatHuggingFace, messages: list[dict[str, str]]) -> str:
    response = model.invoke(messages)

    return clean_code_output(response.content)


if __name__ == "__main__":
    environment = os.environ.get("ENVIRONMENT", "dev")
    language = os.environ.get("LANGUAGE", "java")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

    df = pd.read_csv(f"data/aidev/{language}.csv")
    sample_df = df.sample(n=5)
    dataframe = sample_df if environment == "dev" else df
    dataframe = dataframe[dataframe["code"].str.len() < 20000]
    human_df = dataframe[dataframe["label"] == 1]

    model = get_model(model_name)

    for index, row in human_df.iterrows():
        code_human = row["code"]
        prompt = get_prompt(code_human)
        code_ai = generate_ai_pair(model, prompt)
        human_df.loc[index, "contrast"] = code_ai

    human_df.to_json(
        "data/contrastive-aidev/java_paired.jsonl",
        orient="records",
        lines=True,
        mode="w",
    )
