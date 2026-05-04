"""
=======================================================
SANITY CHECK — Architecture CSLT avec Key-points
=======================================================
Exécutez ce script AVANT de lancer le vrai entraînement.
Tous les tests doivent afficher : ✅ RÉUSSI
=======================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────
# Configuration globale du projet
# ─────────────────────────────────────────────────────
BATCH_SIZE = 4
MAX_FRAMES = 200    # Frames par vidéo après padding
KEYPOINTS  = 1629   # 543 points × 3 coordonnées (x,y,z)
HIDDEN_DIM = 512    # Dimension cachée du Transformer
VOCAB_SIZE = 15000  # Vocabulaire anglais How2Sign
MAX_TEXT   = 50     # Longueur max des phrases (en tokens)
PAD_IDX    = 0      # Index du token <PAD>

# ─────────────────────────────────────────────────────
# Définition du modèle de test (version simplifiée)
# ─────────────────────────────────────────────────────
class MiniCSLTModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Remplace le CNN → reçoit directement les key-points
        self.keypoint_embedding = nn.Sequential(
            nn.Linear(KEYPOINTS, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU()
        )

        # Encodeur Transformer (représente SignJoey Encoder)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=8, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Embedding des tokens textuels pour le décodeur
        self.text_embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM, padding_idx=PAD_IDX)

        # Décodeur Transformer (représente SignJoey Decoder)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=HIDDEN_DIM, nhead=8, batch_first=True, dropout=0.0
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)

        # Classifieur final → remplacé lors du Fine-Tuning
        self.classifier = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, keypoints, target_tokens):
        # keypoints    : [B, T_video, 1629]
        # target_tokens: [B, T_text]

        # 1. Projection des key-points
        x = self.keypoint_embedding(keypoints)          # [B, T_video, 512]

        # 2. Encodage temporel
        memory = self.encoder(x)                        # [B, T_video, 512]

        # 3. Embedding du texte cible
        tgt = self.text_embedding(target_tokens)        # [B, T_text, 512]

        # 4. Décodage avec attention croisée
        out = self.decoder(tgt, memory)                 # [B, T_text, 512]

        # 5. Classification sur le vocabulaire
        logits = self.classifier(out)                   # [B, T_text, 15000]
        return logits


# ═══════════════════════════════════════════════════════
# TEST 1 : FORWARD PASS — Vérification des dimensions
# ═══════════════════════════════════════════════════════
def test_forward_pass():
    print("\n" + "="*55)
    print("TEST 1 : FORWARD PASS — Vérification des dimensions")
    print("="*55)

    model = MiniCSLTModel()
    model.eval()

    fake_keypoints = torch.randn(BATCH_SIZE, MAX_FRAMES, KEYPOINTS)
    fake_labels    = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, MAX_TEXT))

    print(f"  Entrée key-points  : {list(fake_keypoints.shape)}  ← [Batch, Frames, 1629]")

    with torch.no_grad():
        logits = model(fake_keypoints, fake_labels)

    print(f"  Sortie logits      : {list(logits.shape)}  ← [Batch, Mots, Vocabulaire]")

    assert logits.shape == (BATCH_SIZE, MAX_TEXT, VOCAB_SIZE), \
        f"ERREUR : dimension incorrecte ! Attendu [{BATCH_SIZE}, {MAX_TEXT}, {VOCAB_SIZE}]"

    print("\n  ✅ TEST 1 RÉUSSI : Toutes les dimensions sont correctes !")
    return True


# ═══════════════════════════════════════════════════════
# TEST 2 : OVERFIT SUR 1 BATCH — Test de mémorisation
# ═══════════════════════════════════════════════════════
def test_overfit_one_batch():
    print("\n" + "="*55)
    print("TEST 2 : OVERFIT 1 BATCH — Test de mémorisation")
    print("="*55)
    print("  Objectif : la loss DOIT descendre proche de 0.")
    print("  Si elle reste haute → bug dans l'architecture.\n")

    model     = MiniCSLTModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Un seul exemple fixe (toujours le même)
    one_keypoints = torch.randn(1, MAX_FRAMES, KEYPOINTS)
    one_label     = torch.randint(1, VOCAB_SIZE, (1, MAX_TEXT))

    initial_loss = None
    final_loss   = None

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        logits = model(one_keypoints, one_label)
        loss   = criterion(logits.view(-1, VOCAB_SIZE), one_label.view(-1))
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/100  →  Loss = {loss.item():.4f}")

    final_loss = loss.item()
    reduction  = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n  Loss initiale  : {initial_loss:.4f}")
    print(f"  Loss finale    : {final_loss:.4f}")
    print(f"  Réduction      : {reduction:.1f}%")

    if reduction > 80:
        print("\n  ✅ TEST 2 RÉUSSI : Le modèle mémorise bien (>80% de réduction).")
        return True
    else:
        print("\n  ❌ TEST 2 ÉCHOUÉ : La loss ne descend pas assez. Vérifiez lr et architecture.")
        return False


# ═══════════════════════════════════════════════════════
# TEST 3 : GEL DES COUCHES — Vérification du Freeze
# ═══════════════════════════════════════════════════════
def test_freeze_layers():
    print("\n" + "="*55)
    print("TEST 3 : FREEZE — Vérification du gel des couches")
    print("="*55)

    model = MiniCSLTModel()

    # Geler l'encodeur (comme on le fera avec SignJoey pré-entraîné)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Compter les paramètres
    total       = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen      = total - trainable

    print(f"  Paramètres TOTAL          : {total:>12,}")
    print(f"  Paramètres GELÉS ❄️        : {frozen:>12,}  ← Encoder SignJoey")
    print(f"  Paramètres ENTRAÎNABLES 🔥: {trainable:>12,}  ← Decoder + Classifier")
    print(f"  Ratio entraînable         : {100*trainable/total:.1f}%")

    assert frozen > 0, "ERREUR : Aucun paramètre n'est gelé !"
    assert trainable > 0, "ERREUR : Tous les paramètres sont gelés !"

    print("\n  ✅ TEST 3 RÉUSSI : Freeze opérationnel.")
    return True


# ═══════════════════════════════════════════════════════
# TEST 4 : REMPLACEMENT DU CLASSIFIEUR
# ═══════════════════════════════════════════════════════
def test_replace_classifier():
    print("\n" + "="*55)
    print("TEST 4 : REMPLACEMENT du classifieur textuel")
    print("="*55)

    PHOENIX_VOCAB = 3000   # Taille originale PHOENIX-2014T
    HOW2SIGN_VOCAB = 15000 # Votre nouveau vocabulaire

    # Modèle chargé avec l'ancien vocabulaire PHOENIX
    model = MiniCSLTModel()
    # Simule le remplacement comme dans fine_tune_cslt.py
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, HOW2SIGN_VOCAB)

    print(f"  Ancienne couche  : Linear(512 → {PHOENIX_VOCAB})")
    print(f"  Nouvelle couche  : Linear(512 → {HOW2SIGN_VOCAB})")
    print(f"  in_features      : {in_features}")
    print(f"  out_features     : {model.classifier.out_features}")

    assert model.classifier.out_features == HOW2SIGN_VOCAB
    print("\n  ✅ TEST 4 RÉUSSI : Classifieur remplacé correctement.")
    return True


# ═══════════════════════════════════════════════════════
# LANCEMENT DE TOUS LES TESTS
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "★"*55)
    print("  SANITY CHECK — Projet CSLT (Sign Language Translation)")
    print("★"*55)

    results = {
        "Test 1 — Forward Pass"         : test_forward_pass(),
        "Test 2 — Overfit 1 Batch"      : test_overfit_one_batch(),
        "Test 3 — Freeze des couches"   : test_freeze_layers(),
        "Test 4 — Remplacement couche"  : test_replace_classifier(),
    }

    print("\n" + "="*55)
    print("  RÉSUMÉ FINAL")
    print("="*55)
    all_passed = True
    for name, passed in results.items():
        status = "✅ RÉUSSI" if passed else "❌ ÉCHOUÉ"
        print(f"  {status}  →  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 ARCHITECTURE VALIDÉE — Vous pouvez lancer le Fine-Tuning !")
    else:
        print("  ⚠️  Des tests ont échoué — Corrigez avant de continuer.")
    print("="*55 + "\n")
