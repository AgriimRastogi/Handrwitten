import json
import torch
import sys
import asyncio
from pathlib import Path

# Add Handrwitten to sys.path
sys.path.insert(0, str(Path.cwd() / "Handrwitten"))
import numpy as np

from hws.pretrained_model import IAMTokenizer, SynthesisNetwork
from hws.preprocess import offsets_to_coords, normalize_scale, denoise
from playwright.async_api import async_playwright

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

def catmull_rom_to_bezier(pts):
    N = len(pts)
    if N < 2:
        return ""
    if N == 2:
        return f"M {pts[0,0]:.2f},{pts[0,1]:.2f} L {pts[1,0]:.2f},{pts[1,1]:.2f}"
    p = np.vstack([[pts[0]], pts, [pts[-1]]])
    d = [f"M {pts[0,0]:.2f},{pts[0,1]:.2f}"]
    for i in range(1, N):
        p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        d.append(f"C {c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return " ".join(d)

def render_strokes_to_svg(coords_list, filename, titles):
    svg_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 1100" width="900" height="1100">',
        '<rect width="900" height="1100" fill="#fcfcf9"/>'
    ]
    y_cursor = 80
    for coords, title in zip(coords_list, titles):
        svg_parts.append(f'<text x="50" y="{y_cursor-15}" font-family="sans-serif" font-size="14" fill="#888">{title}</text>')
        svg_parts.append(f'<line x1="40" y1="{y_cursor}" x2="860" y2="{y_cursor}" stroke="#c0d0e0" stroke-width="1"/>')
        
        pen_up = coords[:, 2] > 0.5
        stroke_ends = np.where(pen_up)[0]
        start_idx = 0
        min_x = coords[:, 0].min()
        for end_idx in stroke_ends:
            stroke = coords[start_idx:end_idx+1, :2].copy()
            stroke[:, 0] = stroke[:, 0] - min_x + 60
            stroke[:, 1] = y_cursor - stroke[:, 1]
            d = catmull_rom_to_bezier(stroke)
            if d:
                svg_parts.append(f'<path d="{d}" stroke="#16213e" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
            start_idx = end_idx + 1
        y_cursor += 120
    svg_parts.append("</svg>")
    Path(filename).write_text("\n".join(svg_parts), encoding="utf-8")

candidates = []
titles = []
text = "Agriim Rastogi"
c = tokenizer.encode(text + " ").to(device)
u = c.shape[1]

for s in [0, 1, 2, 3, 5, 8, 14, 20]:
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
            sx = sd_out[0, 0, comp] * 0.35
            sy = sd_out[0, 0, comp + num_c] * 0.35
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
            
    deltas = (torch.cat(outputs, dim=0).cpu() * sd + mu).numpy()
    N = len(deltas)
    raw = np.zeros((N, 4), dtype=np.float32)
    raw[:, 0] = deltas[:, 0]
    raw[:, 1] = -deltas[:, 1]
    raw[:, 2] = deltas[:, 2]
    raw[:, 3] = 0.02
    scaled, _ = normalize_scale(raw, target_median=1.0)
    coords = offsets_to_coords(scaled)
    coords = denoise(coords, window=7, polyorder=3)
    base = float(np.percentile(coords[:, 1], 20))
    coords[:, 1] -= base
    coords[:, :2] *= 1.8
    
    candidates.append(coords)
    titles.append(f"Seed {s} (min char steps={char_counts[:len(text)].min()})")

render_strokes_to_svg(candidates, "outputs/agriim_gallery.svg", titles)

async def snap():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 900, "height": 1100})
        await page.goto(f'file:///{Path("outputs/agriim_gallery.svg").resolve().as_posix()}')
        await page.screenshot(path="outputs/agriim_gallery.png")
        await b.close()

asyncio.run(snap())
print("Gallery rendered to outputs/agriim_gallery.png")
