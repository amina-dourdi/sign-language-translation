import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SpatialEmbedding(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.face_stream = nn.Sequential(
            nn.Linear(70 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size // 4)
        )
        self.hand_stream = nn.Sequential(
            nn.Linear(42 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size // 2)
        )
        self.body_stream = nn.Sequential(
            nn.Linear(21 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size // 4)
        )
        self.fusion = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, T, 399)
        face_feat = x[:, :, 0:210]
        hand_feat = x[:, :, 210:336]
        body_feat = x[:, :, 336:399]
        f_out = self.face_stream(face_feat)
        h_out = self.hand_stream(hand_feat)
        b_out = self.body_stream(body_feat)
        combined = torch.cat([f_out, h_out, b_out], dim=-1)
        return self.dropout(self.fusion(combined))

class Recognition(nn.Module):
    def __init__(self, input_dim=399, hidden_size=512, vocab_size=1000):
        super().__init__()
        self.spatial_embedding = SpatialEmbedding(hidden_size)
        self.conv1d = nn.Conv1d(input_dim, hidden_size, kernel_size=3, stride=1, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.visual_head = nn.Linear(hidden_size, vocab_size + 1)

    def forward(self, x):
        x = self.spatial_embedding(x)
        x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        memory = self.encoder(x)
        logits = self.visual_head(memory)
        log_probs = F.log_softmax(logits, dim=-1)
        return {"video_features": memory, "gloss_logits": log_probs}

class VLMapper(nn.Module):
    def __init__(self, in_features=512, out_features=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    def forward(self, visual_outputs):
        x = visual_outputs["video_features"]
        return self.projection(x)

class TranslationNetwork(nn.Module):
    def __init__(self, input_dim=512, vocab_size=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.pos_encoding = PositionalEncoding(input_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.output_layer = nn.Linear(input_dim, vocab_size)

    def forward(self, input_feature, target_ids, tgt_mask=None, memory_mask=None):
        tgt_emb = self.pos_encoding(self.embedding(target_ids))
        output = self.decoder(tgt=tgt_emb, memory=input_feature, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        logits = self.output_layer(output)
        return {"translation_logits": logits}

class SignLanguageModel(nn.Module):
    def __init__(self, input_dim=399, model_dim=512, vocab_size=5000):
        super().__init__()
        # Due to a bug in the original training script, vocab_size wasn't passed to Recognition,
        # so it defaulted to 1000 (making visual_head 1001). We must match this exactly.
        self.recognition_network = Recognition(input_dim=input_dim, hidden_size=model_dim, vocab_size=1000)
        self.vl_mapper = VLMapper(in_features=model_dim, out_features=model_dim)
        self.translation_network = TranslationNetwork(input_dim=model_dim, vocab_size=vocab_size)

    def forward(self, pose_data, target_ids=None, tgt_mask=None, **kwargs):
        pass

    def translate(self, pose_data, max_len=50, rep_penalty=1.3, tokenizer=None):
        self.eval()
        device = pose_data.device
        batch_size = pose_data.size(0)
        with torch.no_grad():
            rec_out = self.recognition_network(pose_data)
            memory = self.vl_mapper(rec_out)
            start_idx = tokenizer.vocab.get("<sos>", 1) if tokenizer else 1
            end_idx = tokenizer.vocab.get("<eos>", 2) if tokenizer else 2
            
            all_generated = []
            for b in range(batch_size):
                mem = memory[b:b+1]
                seq = torch.tensor([start_idx], device=device)
                for _ in range(max_len):
                    out = self.translation_network(mem, seq.unsqueeze(0))
                    logits = out["translation_logits"][:, -1, :]
                    probs = torch.log_softmax(logits, dim=-1)
                    
                    if rep_penalty > 1.0:
                        for tok in seq.unique():
                            if tok.item() > 3:
                                count = (seq == tok).sum().item()
                                probs[0, tok] /= (rep_penalty ** min(count, 3))
                                
                    nxt = probs.argmax(-1)
                    seq = torch.cat([seq, nxt])
                    if nxt.item() == end_idx:
                        break
                all_generated.append(seq)
            
            if tokenizer:
                return [tokenizer.decode(gen.cpu().tolist()) for gen in all_generated]
            return all_generated

class CleanSLTTokenizer:
    def __init__(self):
        self.pad, self.sos, self.eos, self.unk = "<pad>", "<sos>", "<eos>", "<unk>"
        self.vocab = {self.pad: 0, self.sos: 1, self.eos: 2, self.unk: 3}
        self.inv_vocab = {0: self.pad, 1: self.sos, 2: self.eos, 3: self.unk}

    def load_vocab(self, vocab_dict):
        self.vocab = vocab_dict
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
    @property
    def vocab_size(self):
        return len(self.vocab)

    def decode(self, tokens):
        words = [self.inv_vocab.get(t, self.unk) for t in tokens if t > 3 and t != self.vocab.get(self.eos, 2)]
        return " ".join(words)
