import torch

path = "best_finetuned_model.pth"
try:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict) and "model_state_dict" in data:
        state_dict = data["model_state_dict"]
    elif isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
    elif isinstance(data, dict) and "model" in data:
        state_dict = data["model"]
    else:
        state_dict = data

    print("Keys found in checkpoint:", type(data), list(data.keys()) if isinstance(data, dict) else "Not a dict")
    
    first_few_keys = list(state_dict.keys())[:5]
    print("First few keys in state_dict:", first_few_keys)

    if "src_proj.0.weight" in state_dict:
        print("src_proj.0.weight shape:", state_dict["src_proj.0.weight"].shape)
    elif "recognition_network.spatial_embedding.face_stream.0.weight" in state_dict:
        print("Model is SignLanguageModel!")
        print("recognition_network.spatial_embedding.face_stream.0.weight shape:", state_dict["recognition_network.spatial_embedding.face_stream.0.weight"].shape)

except Exception as e:
    print("Error loading:", e)
