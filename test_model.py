import os
import sys

import types
import importlib.util

torchcodec_mock = types.ModuleType("torchcodec")
torchcodec_mock.__spec__ = importlib.util.spec_from_loader("torchcodec", loader=None)
sys.modules["torchcodec"] = torchcodec_mock

# Simulate BATCH environment with just 3 samples
os.environ["ENVIRONMENT"] = "dev"
os.environ["LANGUAGE"] = "java"  # smallest dataset — fastest test
os.environ["MODEL_NAME"] = "google/gemma-4-26B-A4B-it"
os.environ["CHUNK_INDEX"] = "0"
os.environ["TOTAL_CHUNKS"] = "1"

# Just run main.py directly
import main
