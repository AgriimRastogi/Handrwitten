import json
import torch
import sys
import asyncio
from pathlib import Path

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
    if N < 2: return ""
    if N == 2: return f"M {pts[0,0]:.2f},{pts[0,1]:.2f} L {pts[1,0]:.2f},{pts[1,1]:.2f}"
    p = np.vstack([[pts[0]], pts, [pts[-1]]])
    d = [f"M {pts[0,0]:.2f},{pts[0,1]:.2f}"]
    for i in range(1, N):
        p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        d.append(f"C {c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}")
    return " ".join(d)

def generate_line_coords(text, seed, temp=0.35):
    torch.manual_seed(seed)
    np.random.seed(seed)
    c = tokenizer.encode(text + " ").to(device)
    u = c.shape[1]
    
    x = model.get_initial_input().unsqueeze(0)
    w = model.get_initial_window(1)
    k = torch.zeros(1, model.gaussian_components, device=device, dtype=torch.float32)
    hidden1, hidden2, hidden3 = model.get_all_initial_states(1)
    outputs = []
    
    with torch.no_grad():
        for t in range(800):
            x_with_w = torch.cat([x, w], dim=-1)
            h1, hidden1 = model.lstm1(x_with_w, hidden1)
            phi, k = model.window(h1, c, k)
            w = model.window.matmul_3d(phi, c)
            mixture, hidden2, hidden3 = model.compute_mixture(x, h1, w, hidden2, hidden3)
            
            curr_char = phi[0, 0].argmax().item()
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
    
    # Scale: x-height ~16 units, ideal for 40-unit ruled lines
    coords[:, :2] *= 1.7
    return coords

# Let's render a comparison of candidate pages:
combos = [
    ("Combo A (Seeds 27, 35, 43)", 27, 35, 43),
    ("Combo B (Seeds 27, 4, 41)", 27, 4, 41),
    ("Combo C (Seeds 19, 16, 8)", 19, 16, 8),
]

for name, s1, s2, s3 in combos:
    line1 = generate_line_coords("Agriim Rastogi", s1)
    line2 = generate_line_coords("GLA University", s2)
    line3 = generate_line_coords("Project 2026", s3)
    
    # Render notebook page
    width = 650
    height = 500
    margin_left = 75
    margin_top = 80
    line_height = 48
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#fdfdf9"/>',
    ]
    # Ruled lines
    y = margin_top
    while y < height - 30:
        svg.append(f'<line x1="{margin_left - 15}" y1="{y}" x2="{width - 40}" y2="{y}" stroke="#c9d6e3" stroke-width="0.8"/>')
        y += line_height
    # Margin rule
    svg.append(f'<line x1="{margin_left - 20}" y1="0" x2="{margin_left - 20}" y2="{height}" stroke="#e8a0a0" stroke-width="1.2"/>')
    
    # Draw handwriting lines
    for i, coords in enumerate([line1, line2, line3]):
        base_y = margin_top + i * line_height
        min_x = coords[:, 0].min()
        pen_up = coords[:, 2] > 0.5
        stroke_ends = np.where(pen_up)[0]
        start_idx = 0
        for end_idx in stroke_ends:
            stroke = coords[start_idx:end_idx+1, :2].copy()
            stroke[:, 0] = stroke[:, 0] - min_x + margin_left
            stroke[:, 1] = base_y - stroke[:, 1]  # +y was up, SVG +y is down
            d = catmull_rom_to_bezier(stroke)
            if d:
                svg.append(f'<path d="{d}" stroke="#1a1a2e" stroke-width="1.7" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
            start_idx = end_idx + 1
            
    svg.append("</svg>")
    clean_name = name.split()[1] # A, B, C
    out_svg = Path(f"outputs/candidate_{clean_name}.svg")
    out_svg.write_text("\n".join(svg), encoding="utf-8")

async def snap():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for clean_name in ["A", "B", "C"]:
            page = await b.new_page(viewport={"width": 650, "height": 500})
            await page.goto(f'file:///{Path(f"outputs/candidate_{clean_name}.svg").resolve().as_posix()}')
            await page.screenshot(path=f"outputs/candidate_{clean_name}.png")
            await page.close()
        await b.close()

asyncio.run(snap())
print("All candidate screenshots rendered!")
