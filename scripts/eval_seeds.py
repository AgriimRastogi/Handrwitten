import json
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "Handrwitten"))
import numpy as np

from hws.pretrained_model import IAMTokenizer, SynthesisNetwork
from hws.preprocess import offsets_to_coords, normalize_scale, denoise

model_dir = Path("Handrwitten/checkpoints/pretrained")
with open(model_dir / "meta.json") as f:
    meta = json.load(f)

mu = torch.tensor(meta["mu"], dtype=torch.float32)
sd = torch.tensor(meta["std"], dtype=torch.float32)
tokenizer = IAMTokenizer(meta["charset"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SynthesisNetwork.get_default_model(tokenizer.size, device, bias=0.0).to(device)
model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
model.eval()

def evaluate_seeds(text, max_seeds=50, temp=0.35):
    c = tokenizer.encode(text + " ").to(device)
    u = c.shape[1]
    results = []
    
    for s in range(max_seeds):
        torch.manual_seed(s)
        np.random.seed(s)
        
        x = model.get_initial_input().unsqueeze(0)
        w = model.get_initial_window(1)
        k = torch.zeros(1, model.gaussian_components, device=device, dtype=torch.float32)
        hidden1, hidden2, hidden3 = model.get_all_initial_states(1)
        outputs = []
        char_counts = np.zeros(u, dtype=int)
        
        with torch.no_grad():
            for t in range(800):
                x_with_w = torch.cat([x, w], dim=-1)
                h1, hidden1 = model.lstm1(x_with_w, hidden1)
                phi, k = model.window(h1, c, k)
                w = model.window.matmul_3d(phi, c)
                mixture, hidden2, hidden3 = model.compute_mixture(x, h1, w, hidden2, hidden3)
                
                curr_char = phi[0, 0].argmax().item()
                char_counts[curr_char] += 1
                
                pi, mu_out, sd_out, ro, eos = mixture
                comp = torch.multinomial(pi[0, 0], 1).item()
                num_c = len(pi[0, 0])
                mx = mu_out[0, 0, comp]
                my = mu_out[0, 0, comp + num_c]
                sx = sd_out[0, 0, comp] * temp
                sy = sd_out[0, 0, comp + num_c] * temp
                r = ro[0, 0, comp]
                cov = r * sx * sy
                sigma = torch.tensor([[sx**2, cov], [cov, sy**2]])
                gmm = torch.distributions.MultivariateNormal(torch.tensor([mx.item(), my.item()]), sigma)
                v = gmm.sample()
                e = 1.0 if eos[0, 0].item() > 0.5 else 0.0
                x_temp = torch.tensor([v[0].item(), v[1].item(), e], device=device).unsqueeze(0)
                
                last_phi = phi[0, 0, u - 1].item()
                is_eos = x_temp[0, 2].item() > 0.5
                if last_phi > 0.8 or (curr_char == u - 1 and is_eos):
                    x_temp[0, 2] = 1.0
                    outputs.append(x_temp)
                    break
                outputs.append(x_temp)
                x = x_temp.unsqueeze(0)
                
        valid_counts = char_counts[:len(text)]
        min_c = int(valid_counts.min())
        total_pts = len(outputs)
        
        # Check if unbounded wild loop (wild loops have >600 points or huge variance)
        deltas = (torch.cat(outputs, dim=0).cpu() * sd + mu).numpy()
        N = len(deltas)
        raw = np.zeros((N, 4), dtype=np.float32)
        raw[:, 0] = deltas[:, 0]
        raw[:, 1] = -deltas[:, 1]
        raw[:, 2] = deltas[:, 2]
        raw[:, 3] = 0.02
        scaled, _ = normalize_scale(raw, target_median=1.0)
        coords = offsets_to_coords(scaled)
        
        # Bounding box sanity check
        w_span = coords[:, 0].max() - coords[:, 0].min()
        h_span = coords[:, 1].max() - coords[:, 1].min()
        aspect = w_span / (h_span + 1e-5)
        
        # Good candidate: all characters attended (min_c >= 5), reasonable points (200-550), aspect > 2.0 (horizontal line)
        is_good = (min_c >= 5) and (150 <= total_pts <= 550) and (aspect > 2.0)
        results.append({
            "seed": s,
            "min_c": min_c,
            "pts": total_pts,
            "aspect": aspect,
            "is_good": is_good,
            "coords": coords,
            "counts": valid_counts.tolist(),
        })
        
    good_results = [r for r in results if r["is_good"]]
    good_results.sort(key=lambda r: -r["min_c"])
    print(f"Text: '{text}' -> Found {len(good_results)}/{max_seeds} good seeds")
    for r in good_results[:5]:
        print(f"  Seed {r['seed']:2d}: min_steps={r['min_c']}, total_pts={r['pts']}, aspect={r['aspect']:.1f}")
    return good_results

print("--- Agriim Rastogi ---")
res_agriim = evaluate_seeds("Agriim Rastogi", 50)

print("\n--- GLA University ---")
res_gla = evaluate_seeds("GLA University", 50)

print("\n--- Project 2026 ---")
res_proj = evaluate_seeds("Project 2026", 50)
