"""Debug slashes attention."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Handrwitten"))

import torch
import numpy as np
from hws.pretrained_model import PretrainedGravesSynthesis

m = PretrainedGravesSynthesis(REPO_ROOT / "Handrwitten" / "checkpoints" / "pretrained", bias=1.0)
clean_text = "/ / / / /"
c = m.tokenizer.encode(clean_text + " ").to(m.device)
u = c.shape[1]

print(f"Encoded u={u}")

torch.manual_seed(42)
np.random.seed(42)

x = m.model.get_initial_input().unsqueeze(0)
w = m.model.get_initial_window(1)
k = torch.zeros(1, m.model.gaussian_components, device=m.device, dtype=torch.float32)
hidden1, hidden2, hidden3 = m.model.get_all_initial_states(1)

for t in range(250):
    x_with_w = torch.cat([x, w], dim=-1)
    h1, hidden1 = m.model.lstm1(x_with_w, hidden1)
    phi, k = m.model.window(h1, c, k)
    curr_char = int(phi[0, 0].argmax().item())
    w = m.model.window.matmul_3d(phi, c)
    mixture, hidden2, hidden3 = m.model.compute_mixture(x, h1, w, hidden2, hidden3)
    pi, mu_out, sd_out, ro, eos = mixture
    
    char_name = clean_text[curr_char] if curr_char < len(clean_text) else "END"
    if t % 10 == 0:
        print(f"t={t:3d}: curr_char={curr_char:2d} ('{char_name}'), k_mean={k.mean().item():.2f}, eos={eos[0,0].item():.2f}")
    if curr_char >= u - 1 and eos[0, 0].item() > 0.5:
        print(f"Done at t={t}")
        break
    x = torch.zeros(1, 1, 3, device=m.device)
