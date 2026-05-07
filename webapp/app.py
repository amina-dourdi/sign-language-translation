import os
import sys
import json
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Ajouter le dossier parent au PATH pour importer le modèle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.cslt_model import CSLTModel
from data_pipeline.tokenizer import Tokenizer

app = FastAPI(title="CSLT Live Translation")

# 1. Configuration du modèle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / "best_model_how2sign.pth")
TOKENIZER_PATH = str(PROJECT_ROOT / "data" / "processed" / "tokenizer.json")

print("Chargement du tokenizer...")
tokenizer = Tokenizer()
if os.path.exists(TOKENIZER_PATH):
    tokenizer.load(TOKENIZER_PATH)
else:
    print("⚠️ Attention: Tokenizer non trouvé. Le modèle renverra des indices au lieu de mots.")

print("Chargement du modèle...")
# Note: Ces dimensions doivent correspondre à ton entraînement
# Le modèle est initialisé sans charger les poids si le fichier n'existe pas encore
model = CSLTModel(
    input_dim=411,
    vocab_size=tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 10000,
    d_model=512,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,  # Correction ici !
    max_src_len=150,       # Correction ici !
    max_tgt_len=80         # Correction ici !
).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load(MODEL_PATH, device=DEVICE)
    print("✅ Modèle chargé avec succès !")
else:
    print("⚠️ Attention: Fichier du modèle non trouvé, utilisation d'un modèle non entraîné pour la démo.")

model.eval()

# 2. Servir les fichiers statiques (le Frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# 3. WebSocket pour la traduction en temps réel
@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Nouvelle connexion WebSocket établie.")
    
    frames_buffer = []
    MAX_FRAMES = 150 # Doit correspondre à ton max_frames de l'entraînement
    
    try:
        while True:
            # Recevoir les points clés (411 valeurs) depuis le navigateur
            data = await websocket.receive_text()
            keypoints = json.loads(data)
            
            if len(keypoints) == 411:
                frames_buffer.append(keypoints)
                
                # Si on a accumulé assez de frames (ex: 30 frames pour commencer à traduire)
                # Ou on garde une fenêtre glissante
                if len(frames_buffer) >= 30:
                    # Préparer le tenseur
                    input_seq = np.array(frames_buffer, dtype=np.float32)
                    
                    # Normalisation Z-score (comme dans le preprocessing)
                    mean = np.mean(input_seq, axis=0, keepdims=True)
                    std = np.std(input_seq, axis=0, keepdims=True)
                    input_seq = (input_seq - mean) / (std + 1e-8)
                    
                    # Padding si nécessaire
                    if len(input_seq) < MAX_FRAMES:
                        padding = np.zeros((MAX_FRAMES - len(input_seq), 411), dtype=np.float32)
                        input_seq = np.concatenate([input_seq, padding], axis=0)
                    else:
                        input_seq = input_seq[:MAX_FRAMES]
                        
                    # Inférence PyTorch
                    tensor_input = torch.FloatTensor(input_seq).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        # Utiliser greedy decoding (translate)
                        try:
                            generated = model.translate(tensor_input, max_len=15, tokenizer=tokenizer if os.path.exists(TOKENIZER_PATH) else None)
                            
                            # Renvoyer la traduction
                            if isinstance(generated, list):
                                translation = generated[0]
                            else:
                                translation = " ".join([str(idx) for idx in generated[0].tolist()])
                                
                            await websocket.send_json({"translation": translation})
                        except Exception as e:
                            print(f"Erreur d'inférence: {e}")
                    
                    # Pour éviter que la mémoire tampon ne grandisse indéfiniment, 
                    # on garde une fenêtre glissante (on enlève les plus vieilles frames)
                    frames_buffer = frames_buffer[5:] # Avance de 5 frames
                    
    except WebSocketDisconnect:
        print("Client déconnecté.")
    except Exception as e:
        print(f"Erreur WebSocket: {e}")
