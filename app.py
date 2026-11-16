import re
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- 1. Tokenizer (Matches Training Pipeline) ---
def tokenize(text):
    text = text.strip().lower()
    return re.findall(r"[a-z0-9]+(?:[.,'-][a-z0-9]+)*|[^\w\s]", text)

# --- 2. MiniGPT Transformer Architecture ---
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class BikeMiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=3, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate_answer(self, question, stoi, itos, max_new_tokens=95):
        self.eval()
        device = next(self.parameters()).device
        clean_q = question.strip().rstrip("?!.")
        q_ids = [stoi.get(t, stoi["<UNK>"]) for t in tokenize(clean_q)]
        idx = torch.tensor([[stoi["<BOS>"]] + q_ids + [stoi["<SEP>"]]], dtype=torch.long, device=device)

        generated = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = next_id.item()
            if tok_id == stoi["<EOS>"]:
                break
            generated.append(itos[tok_id])
            idx = torch.cat([idx, next_id], dim=1)

        text = " ".join(generated)
        text = re.sub(r"\s+([.,;:!?)\]])", r"\1", text)
        text = re.sub(r"([(\[])\s+", r"\1", text)
        return text, len(generated)

# --- 3. Load Saved Model & Dataset on Startup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("bike_minigpt.pt", map_location=device)
cfg = checkpoint["config"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

model = BikeMiniGPT(**cfg).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
total_params = sum(p.numel() for p in model.parameters())

df = pd.read_csv("motorcycle_slm_dataset.csv")

# --- 4. Web Routes & REST API ---
@app.route("/")
def home():
    return render_template("index.html", total_params=f"{total_params:,}", bike_count=len(df))

@app.route("/api/bikes", methods=["GET"])
def get_bikes():
    bikes = df[["bike_id", "brand", "model", "category", "engine_cc", "ex_showroom_price_inr"]].to_dict(orient="records")
    return jsonify(bikes)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    t0 = time.perf_counter()
    answer, token_count = model.generate_answer(prompt, stoi, itos)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    return jsonify({
        "prompt": prompt,
        "response": answer,
        "tokens": token_count,
        "latency_ms": latency_ms
    })

if __name__ == "__main__":
    print(f"BikeMiniGPT Backend running on http://127.0.0.1:5000 | Device: {device.upper()}")
    app.run(debug=True, host="127.0.0.1", port=5000)