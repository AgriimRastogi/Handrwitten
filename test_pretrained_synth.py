import json
from collections import defaultdict
import torch
import numpy as np
import sys
from pathlib import Path

# Add Handrwitten to sys.path so hws can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent / "Handrwitten"))

from xray_models import SynthesisNetwork
from hws.render import render_page, InkStyle, PageLayout

# 1. Clean self-contained Tokenizer matching meta.json
class Tokenizer:
    def __init__(self, charset):
        self._charset = str(charset)
        self._chr_to_int = defaultdict(int)
        self._int_to_chr = {}
        for i, ch in enumerate(self._charset):
            self._int_to_chr[i + 1] = ch
            self._chr_to_int[ch] = i + 1

    @property
    def size(self):
        return len(self._charset) + 1

    def tokenize(self, s):
        return [self._chr_to_int[ch] for ch in s]


def transcriptions_to_tensor(tokenizer, transcriptions):
    eye = torch.eye(tokenizer.size)
    token_sequences = []
    for s in transcriptions:
        tokens = tokenizer.tokenize(s)
        token_sequences.append(eye[tokens].numpy().tolist())
    return torch.tensor(token_sequences, dtype=torch.float32)


# 2. Load metadata & weights
meta_path = "Handrwitten/checkpoints/pretrained/meta.json"
with open(meta_path, "r") as f:
    meta = json.load(f)

mu = torch.tensor(meta["mu"], dtype=torch.float32)
sd = torch.tensor(meta["std"], dtype=torch.float32)
charset = meta["charset"]
tokenizer = Tokenizer(charset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bias = 1.2

model = SynthesisNetwork.get_default_model(tokenizer.size, device, bias=bias).to(device)
state_dict = torch.load("Handrwitten/checkpoints/pretrained/model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()


def generate_line(text: str, bias_val: float = 1.2, max_steps: int = 1200) -> np.ndarray:
    model.mixture.bias = bias_val
    sentinel = " "
    full_text = text + sentinel
    c = transcriptions_to_tensor(tokenizer, [full_text]).to(device)

    with torch.no_grad():
        sampled_handwriting, phi = model.sample_means_with_attention(
            context=c, steps=max_steps, stochastic=True
        )

    # Undo normalization: deltas = tensor * sd + mu
    deltas = (sampled_handwriting.cpu() * sd + mu).numpy()  # (N, 3): [dx, dy, eos]
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    eos = deltas[:, 2]

    # Convert to integrated coordinates: [x, y, pen_up, time]
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    time_pts = np.arange(len(x), dtype=np.float32) * 0.02

    pts = np.stack([x, y, eos, time_pts], axis=-1).astype(np.float32)
    return pts


prompts = ["Agriim Rastogi", "GLA University", "Project 2026"]
lines = []

print(f"Device: {device}")
for p in prompts:
    pts = generate_line(p, bias_val=1.2)
    lines.append(pts)
    print(f"Synthesized '{p}': {len(pts)} points, bounds: x=[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}], y=[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")

layout = PageLayout(ruled=True, margin_rule=True)
style = InkStyle(color="#1a1a2e", width=1.6)
page_svg = render_page(lines, layout=layout, style=style)

out_file = Path("outputs/pretrained_my_notes.svg")
out_file.write_text(page_svg, encoding="utf-8")
print(f"Saved full document to {out_file} ({len(page_svg)} bytes)")
