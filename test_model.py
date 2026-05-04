from transformers import AutoProcessor, AutoModelForCausalLM
import sys
from unittest.mock import MagicMock

# Mock torchcodec since it's not available on Compute Canada
sys.modules["torchcodec"] = MagicMock()

model_name = "google/gemma-4-26B-A4B-it"

print("Step 1: Loading processor...")
processor = AutoProcessor.from_pretrained(model_name)
print("Processor loaded OK")

print("Step 2: Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)
print("Model loaded OK")

print("Step 3: Running inference...")
messages = [
    {"role": "system", "content": "You are a code generator. Output only raw code."},
    {"role": "user", "content": "Write a hello world in python."},
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
result = processor.parse_response(response)

print("Inference OK")
print("Output:", result)
