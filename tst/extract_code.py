import json
import os

notebook_path = "MSKA_SignLanguage_Translation_v1_OUMAIMA (1).ipynb"
output_path = "train_colab.py"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

extracted_code = []

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "class CSLTModel" in source or "class Tokenizer" in source or "import torch" in source:
            extracted_code.append(source)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(extracted_code))

print(f"Extracted to {output_path}")
