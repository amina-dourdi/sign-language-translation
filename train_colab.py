"""
=============================================================
train_colab.py — IMPROVED Training Script v3
=============================================================
Fixes from v2:
  1. BUG FIX: val/test shared augmented dataset (corrupted eval)
  2. Added 1D Temporal CNN before Transformer (captures local motion)
  3. Better augmentation: temporal warping + joint masking
  4. Tuned hyperparameters for small dataset
  5. Checkpoint resume for Colab

USAGE:
  python train_colab.py
  python train_colab.py --resume
=============================================================
"""

import os, sys, json, time, math, re, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = {
    "metadata_path": str(PROJECT_ROOT / "data" / "processed" / "metadata.json"),
    "tokenizer_path": str(PROJECT_ROOT / "data" / "processed" / "tokenizer.json"),
    "checkpoint_dir": str(PROJECT_ROOT / "checkpoints"),
    # Model
    "input_dim": 411,
    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.25,
    "max_frames": 150,
    "max_seq_len": 80,
    # Training
    "epochs": 60,
    "batch_size": 8,
    "accumulation_steps": 4,
    "lr": 5e-4,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,
    "max_vocab_size": 10000,
    "patience": 999,            # Disabled — train all epochs
    "warmup_steps": 300,
    # Augmentation
    "noise_std": 0.03,
    "frame_drop_prob": 0.15,
    "scale_range": (0.85, 1.15),
    "joint_mask_prob": 0.1,
    "time_warp_prob": 0.3,
    # Decoding
    "repetition_penalty": 1.3,
    # System
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ═══════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════
class Tokenizer:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3
    def __init__(self, max_vocab_size=10000, max_seq_len=80):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4

    @staticmethod
    def clean(s):
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9\s']", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def build_vocab(self, sentences):
        counter = Counter()
        for s in sentences:
            counter.update(self.clean(s).split())
        for word, _ in counter.most_common(self.max_vocab_size - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.vocab_size = len(self.word2idx)

    def encode(self, sentence):
        tokens = [self.SOS]
        for w in self.clean(sentence).split():
            tokens.append(self.word2idx.get(w, self.UNK))
        tokens.append(self.EOS)
        return tokens[:self.max_seq_len]

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.EOS: break
            if idx in (self.PAD, self.SOS): continue
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)

    def pad_sequence(self, tokens):
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [self.PAD] * (self.max_seq_len - len(tokens))
        return tokens[:self.max_seq_len]

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({"word2idx": self.word2idx, "max_seq_len": self.max_seq_len}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.max_seq_len = data.get("max_seq_len", 80)
        self.vocab_size = len(self.word2idx)
        print(f"  Tokenizer loaded: vocab_size={self.vocab_size}")


# ═══════════════════════════════════════════════
# DATA AUGMENTATION (stronger)
# ═══════════════════════════════════════════════
class KeypointAugmentation:
    def __init__(self, cfg):
        self.noise_std = cfg["noise_std"]
        self.frame_drop = cfg["frame_drop_prob"]
        self.scale_range = cfg["scale_range"]
        self.joint_mask = cfg["joint_mask_prob"]
        self.time_warp = cfg["time_warp_prob"]

    def __call__(self, kp):
        # Gaussian noise
        kp = kp + torch.randn_like(kp) * self.noise_std
        # Random frame dropout
        if self.frame_drop > 0:
            mask = (torch.rand(kp.size(0)) > self.frame_drop).float().unsqueeze(1)
            kp = kp * mask
        # Random scale
        lo, hi = self.scale_range
        kp = kp * torch.empty(1).uniform_(lo, hi).item()
        # Joint masking (zero out random feature groups)
        if self.joint_mask > 0:
            feat_mask = (torch.rand(kp.size(1)) > self.joint_mask).float()
            kp = kp * feat_mask.unsqueeze(0)
        # Temporal warping (slight speed variation via interpolation)
        if torch.rand(1).item() < self.time_warp:
            T = kp.size(0)
            speed = torch.empty(1).uniform_(0.85, 1.15).item()
            new_T = min(int(T * speed), T)
            indices = torch.linspace(0, T - 1, new_T).long()
            warped = kp[indices]
            if warped.size(0) < T:
                pad = torch.zeros(T - warped.size(0), kp.size(1))
                warped = torch.cat([warped, pad], 0)
            kp = warped[:T]
        return kp


# ═══════════════════════════════════════════════
# DATASET (separate train/val instances)
# ═══════════════════════════════════════════════
class CSLTDataset(Dataset):
    def __init__(self, metadata_path, tokenizer, max_frames=150, augment=None):
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.augment = augment
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        self.samples = []
        # On force le chemin vers le VRAI dossier Colab qu'on vient de vérifier
        colab_kp_dir = "/content/data/keypoints"
        
        for clip_id, info in metadata.items():
            # 🐛 LE FIX EST ICI : On coupe par les slashs ET les antislashs !
            raw_path = info.get("npy_path", f"{clip_id}.npy")
            filename = raw_path.replace("\\", "/").split("/")[-1]
            
            forced_path = os.path.join(colab_kp_dir, filename)
            
            sentence = info.get("sentence", "unknown") # Au cas où c'est manquant
            
            if os.path.exists(forced_path):
                self.samples.append({
                    "npy_path": forced_path, 
                    "sentence": sentence,
                    "original_frames": info.get("original_frames", max_frames)
                })
        print(f"  [Dataset] {len(self.samples)} samples (aug={'ON' if augment else 'OFF'})")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        kp = torch.FloatTensor(np.load(s["npy_path"]))
        if self.augment:
            kp = self.augment(kp)
        # On gère le cas où la phrase est "unknown" pour éviter le crash
        tokens = self.tokenizer.pad_sequence(self.tokenizer.encode(s["sentence"]))
        return kp, torch.LongTensor(tokens), s["original_frames"], len(tokens)


# ═══════════════════════════════════════════════
# INDEXED SUBSET (to split without sharing dataset)
# ═══════════════════════════════════════════════
class SubsetByIndices(Dataset):
    """Wraps a dataset with specific indices - no shared references."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.dataset[self.indices[idx]]


# ═══════════════════════════════════════════════
# MODEL with Temporal CNN
# ═══════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalCNN(nn.Module):
    """1D CNN to capture local motion patterns before the Transformer."""
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D] -> conv needs [B, D, T]
        residual = x
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(out + residual)


class CSLTModel(nn.Module):
    def __init__(self, input_dim=411, vocab_size=10000, d_model=256,
                 nhead=4, enc_layers=3, dec_layers=3,
                 dim_ff=512, dropout=0.25, max_src=150, max_tgt=80):
        super().__init__()
        self.d_model = d_model
        # Encoder with Temporal CNN
        self.src_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.temporal_cnn = TemporalCNN(d_model, dropout)
        self.src_pe = PositionalEncoding(d_model, max_src, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers,
                                             enable_nested_tensor=False)
        # Decoder
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_pe = PositionalEncoding(d_model, max_tgt, dropout)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        src_mask = (src.sum(-1) == 0)
        tgt_in = tgt[:, :-1]
        tgt_pad = (tgt_in == 0)
        # Encode with temporal CNN
        x = self.src_proj(src)
        x = self.temporal_cnn(x)
        mem = self.encoder(self.src_pe(x), src_key_padding_mask=src_mask)
        # Decode
        causal = torch.triu(torch.ones(tgt_in.size(1), tgt_in.size(1),
                            device=src.device), diagonal=1).bool()
        emb = self.tgt_pe(self.tgt_emb(tgt_in) * (self.d_model ** 0.5))
        dec = self.decoder(emb, mem, tgt_mask=causal,
                           tgt_key_padding_mask=tgt_pad,
                           memory_key_padding_mask=src_mask)
        return self.out_proj(dec)

    @torch.no_grad()
    def translate(self, src, max_len=80, rep_penalty=1.3, tokenizer=None):
        self.eval()
        dev = src.device; B = src.size(0)
        src_mask = (src.sum(-1) == 0)
        x = self.src_proj(src)
        x = self.temporal_cnn(x)
        mem = self.encoder(self.src_pe(x), src_key_padding_mask=src_mask)
        gen = torch.full((B, 1), 1, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len):
            causal = torch.triu(torch.ones(gen.size(1), gen.size(1), device=dev), 1).bool()
            emb = self.tgt_pe(self.tgt_emb(gen) * (self.d_model ** 0.5))
            dec = self.decoder(emb, mem, tgt_mask=causal, memory_key_padding_mask=src_mask)
            logits = self.out_proj(dec[:, -1, :])
            if rep_penalty > 1.0:
                for b in range(B):
                    for tok in gen[b].unique():
                        if tok.item() > 3:
                            count = (gen[b] == tok).sum().item()
                            logits[b, tok] /= (rep_penalty ** min(count, 3))
            nxt = logits.argmax(-1)
            nxt[done] = 0
            gen = torch.cat([gen, nxt.unsqueeze(1)], 1)
            done = done | (nxt == 2)
            if done.all(): break
        if tokenizer:
            return [tokenizer.decode(gen[i].cpu().tolist()) for i in range(B)]
        return gen

    def count_params(self):
        t = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Parameters: {tr:,} trainable / {t:,} total")
        return tr, t


# ═══════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup = warmup_steps
        self.total = total_steps
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            scale = self.step_num / max(1, self.warmup)
        else:
            progress = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = blr * max(scale, 1e-7)

    def get_lr(self): return self.optimizer.param_groups[0]['lr']


# ═══════════════════════════════════════════════
# BLEU
# ═══════════════════════════════════════════════
def compute_bleu(preds, refs, max_n=4):
    def tokenize(s): return re.sub(r"[^a-z0-9\s']", " ", s.lower()).split()
    def ngrams(toks, n):
        ng = {}
        for i in range(len(toks) - n + 1):
            g = tuple(toks[i:i+n]); ng[g] = ng.get(g, 0) + 1
        return ng
    clip = [0]*max_n; tot = [0]*max_n; pl = rl = 0
    for p, r in zip(preds, refs):
        pt, rt = tokenize(p), tokenize(r)
        pl += len(pt); rl += len(rt)
        for n in range(1, max_n+1):
            pn, rn = ngrams(pt, n), ngrams(rt, n)
            for g, c in pn.items():
                clip[n-1] += min(c, rn.get(g, 0)); tot[n-1] += c
    bp = 1.0 if pl >= rl else (math.exp(1 - rl/pl) if pl > 0 else 0)
    res = {"bp": bp}
    for n in range(1, max_n+1):
        precs = [clip[i]/tot[i] if tot[i] > 0 else 0 for i in range(n)]
        if all(p > 0 for p in precs):
            res[f"bleu{n}"] = round(bp * math.exp(sum(math.log(p) for p in precs)/n) * 100, 2)
        else: res[f"bleu{n}"] = 0.0
    return res


def collate_fn(batch):
    kp, tgt, sl, tl = zip(*batch)
    return torch.stack(kp), torch.stack(tgt), torch.LongTensor(sl), torch.LongTensor(tl)


# ═══════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, history):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler_step": scheduler.step_num, "epoch": epoch,
                "best_val_loss": best_val, "history": history}, path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.step_num = ckpt["scheduler_step"]
    print(f"  📂 Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["history"]


# ═══════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════
def train(resume=False):
    print("=" * 60)
    print("   CSLT TRAINING v3 (Temporal CNN + Fixed Augmentation)")
    print("=" * 60)
    print(f"  Device: {CONFIG['device']}")
    device = torch.device(CONFIG["device"])

    # Tokenizer
    tok = Tokenizer(CONFIG["max_vocab_size"], CONFIG["max_seq_len"])
    if os.path.exists(CONFIG["tokenizer_path"]):
        tok.load(CONFIG["tokenizer_path"])
    else:
        with open(CONFIG["metadata_path"]) as f:
            meta = json.load(f)
        tok.build_vocab([v["sentence"] for v in meta.values() if v.get("sentence")])
        tok.save(CONFIG["tokenizer_path"])

    # ── FIX: Create SEPARATE datasets for train (augmented) and val/test (clean) ──
    aug = KeypointAugmentation(CONFIG)
    train_ds_full = CSLTDataset(CONFIG["metadata_path"], tok, CONFIG["max_frames"], augment=aug)
    eval_ds_full = CSLTDataset(CONFIG["metadata_path"], tok, CONFIG["max_frames"], augment=None)

    # Split indices (same seed = same split)
    n = len(train_ds_full)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    tr_n = int(n * 0.8); va_n = int(n * 0.1)
    tr_idx = indices[:tr_n]
    va_idx = indices[tr_n:tr_n + va_n]
    te_idx = indices[tr_n + va_n:]

    # Train uses augmented dataset, val/test use clean dataset
    tr_subset = SubsetByIndices(train_ds_full, tr_idx)
    va_subset = SubsetByIndices(eval_ds_full, va_idx)
    te_subset = SubsetByIndices(eval_ds_full, te_idx)

    tr_loader = DataLoader(tr_subset, CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=CONFIG["num_workers"])
    va_loader = DataLoader(va_subset, CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=CONFIG["num_workers"])
    te_loader = DataLoader(te_subset, CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=CONFIG["num_workers"])
    print(f"  Train: {len(tr_idx)} (augmented) | Val: {len(va_idx)} (clean) | Test: {len(te_idx)} (clean)")

    # Model
    model = CSLTModel(
        CONFIG["input_dim"], tok.vocab_size, CONFIG["d_model"], CONFIG["nhead"],
        CONFIG["num_encoder_layers"], CONFIG["num_decoder_layers"],
        CONFIG["dim_feedforward"], CONFIG["dropout"],
        CONFIG["max_frames"], CONFIG["max_seq_len"]
    ).to(device)
    model.count_params()

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    total_steps = CONFIG["epochs"] * len(tr_loader) // CONFIG["accumulation_steps"]
    scheduler = WarmupCosineScheduler(optimizer, CONFIG["warmup_steps"], total_steps)

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "checkpoint_v3.pth")
    best_path = os.path.join(CONFIG["checkpoint_dir"], "best_model_v3.pth")
    start_epoch = 1; best_val = float('inf')
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if resume and os.path.exists(ckpt_path):
        start_epoch, best_val, history = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        start_epoch += 1

    patience_ctr = 0
    accum = CONFIG["accumulation_steps"]

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        model.train()
        t0 = time.time()
        ep_loss = ep_acc = ep_n = 0
        optimizer.zero_grad()

        for i, (kp, tgt, _, _) in enumerate(tr_loader):
            kp, tgt = kp.to(device), tgt.to(device)
            logits = model(kp, tgt)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)) / accum
            loss.backward()

            if (i + 1) % accum == 0 or (i + 1) == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            ep_loss += loss.item() * accum
            mask = (tgt[:, 1:] != 0)
            ep_acc += ((logits.argmax(-1) == tgt[:, 1:]) & mask).sum().item()
            ep_n += mask.sum().item()

        tr_loss = ep_loss / len(tr_loader)
        tr_acc = ep_acc / max(ep_n, 1) * 100

        # Validate
        model.eval(); vl = va = vn = 0
        with torch.no_grad():
            for kp, tgt, _, _ in va_loader:
                kp, tgt = kp.to(device), tgt.to(device)
                logits = model(kp, tgt)
                vl += criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)).item()
                mask = (tgt[:, 1:] != 0)
                va += ((logits.argmax(-1) == tgt[:, 1:]) & mask).sum().item()
                vn += mask.sum().item()
        val_loss = vl / max(len(va_loader), 1)
        val_acc = va / max(vn, 1) * 100

        elapsed = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"  Ep {epoch:3d}/{CONFIG['epochs']} │ TrL:{tr_loss:.3f} VaL:{val_loss:.3f} │ "
              f"TrA:{tr_acc:.1f}% VaA:{val_acc:.1f}% │ LR:{scheduler.get_lr():.2e} │ {elapsed:.0f}s")

        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val, history)

        if val_loss < best_val:
            best_val = val_loss; patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Best (val_loss={val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"\n  ⏹️  Early stopping at epoch {epoch}"); break

    # Final eval
    print("\n" + "─" * 50 + "\n  FINAL EVALUATION\n" + "─" * 50)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    all_preds, all_refs = [], []
    with torch.no_grad():
        for kp, tgt, _, _ in te_loader:
            preds = model.translate(kp.to(device), CONFIG["max_seq_len"], CONFIG["repetition_penalty"], tok)
            all_preds.extend(preds)
            for i in range(tgt.size(0)):
                all_refs.append(tok.decode(tgt[i].tolist()))

    scores = compute_bleu(all_preds, all_refs)
    print(f"\n  BLEU-1: {scores['bleu1']}  BLEU-2: {scores['bleu2']}  "
          f"BLEU-3: {scores['bleu3']}  BLEU-4: {scores['bleu4']}  BP: {scores['bp']:.4f}")
    for i in range(min(5, len(all_preds))):
        print(f"\n  [REF] {all_refs[i]}\n  [PRD] {all_preds[i]}")

    with open(os.path.join(CONFIG["checkpoint_dir"], "history_v3.json"), 'w') as f:
        json.dump(history, f, indent=2)
    print("\n  ✅ TRAINING COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)
