import json
import os

path = "data/processed/tokenizer.json"
print("Exists?", os.path.exists(path))
if os.path.exists(path):
    with open(path) as f:
        data = json.load(f)
    print("Vocab length:", len(data.get("word2idx", {})))
