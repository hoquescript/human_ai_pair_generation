import os
import runpy

# Simulate BATCH environment with just 5 samples
os.environ["ENVIRONMENT"] = "dev"
os.environ["LANGUAGE"] = "java"
os.environ["MODEL_NAME"] = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
os.environ["CHUNK_INDEX"] = "0"
os.environ["TOTAL_CHUNKS"] = "1"

runpy.run_path("main.py", run_name="__main__")
