"""Test script for Character-Adaptive Dwell Control & Normalization."""

import sys
from pathlib import Path

# Add Handrwitten to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Handrwitten"))

import numpy as np
import torch
from hws.pretrained_model import PretrainedGravesSynthesis, normalize_scale, offsets_to_coords, denoise
from hws.render import render_line, InkStyle
from playwright.sync_api import sync_playwright


def get_adaptive_dwell(ch: str) -> int:
    """Return character-specific dwell ceiling."""
    if ch in " .,;:!?-\"/()[]'":
        return 12
    elif ch.isdigit():
        return 18
    elif ch.isupper():
        return 22
    elif ch in "mwMWQ":
        return 34
    else:
        return 28


def sanitize_and_normalize(text: str) -> str:
    """Normalize repeated non-alphanumeric punctuation and unsupported tokens."""
    import re
    # If there are repeated identical punctuation marks (like ///// or -----), insert a space between them
    # so the RNN treats each as an independent stroke rather than an impossible cursive ligature
    def space_repeats(match):
        s = match.group(0)
        return " ".join(list(s))

    # Pattern for 2+ consecutive identical punctuation characters
    text = re.sub(r'([/\\#\-\*\=\_\+])\1+', space_repeats, text)
    return text.strip()


def generate_adaptive(model: PretrainedGravesSynthesis, text: str, seed: int = 42, bias: float = 1.0):
    clean_text = sanitize_and_normalize(text)
    if not clean_text:
        return np.zeros((0, 4), dtype=np.float32)

    if hasattr(model.model, "mixture"):
        model.model.mixture.bias = bias

    # Encode with trailing sentinel space
    c = model.tokenizer.encode(clean_text + " ").to(model.device)
    u = c.shape[1]

    torch.manual_seed(seed)
    np.random.seed(seed)

    x = model.model.get_initial_input().unsqueeze(0)
    w = model.model.get_initial_window(1)
    k = torch.zeros(1, model.model.gaussian_components, device=model.device, dtype=torch.float32)
    hidden1, hidden2, hidden3 = model.model.get_all_initial_states(1)

    outputs = []
    char_counts = np.zeros(u, dtype=int)
    curr_char = 0
    prev_char = 0
    dwell = 0

    full_str = clean_text + " "

    with torch.no_grad():
        for t in range(1200):
            x_with_w = torch.cat([x, w], dim=-1)
            h1, hidden1 = model.model.lstm1(x_with_w, hidden1)
            phi, k = model.model.window(h1, c, k)

            curr_char = int(phi[0, 0].argmax().item())
            char_counts[curr_char] += 1

            if curr_char == prev_char:
                dwell += 1
            else:
                prev_char = curr_char
                dwell = 0

            # Dynamic adaptive dwell based on current attended character
            active_ch = full_str[min(curr_char, len(full_str) - 1)]
            max_dwell = get_adaptive_dwell(active_ch)

            # Monotonic nudge when dwell exceeds ceiling
            if dwell > max_dwell and curr_char < u - 1:
                k = k + 0.8
                dwell = 0

            w = model.model.window.matmul_3d(phi, c)
            mixture, hidden2, hidden3 = model.model.compute_mixture(x, h1, w, hidden2, hidden3)

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
            x_temp = torch.tensor([v[0].item(), v[1].item(), e], device=model.device).unsqueeze(0)

            last_phi = phi[0, 0, u - 1].item()
            min_pts = max(40, len(clean_text) * 12)
            if (last_phi > 0.6 or curr_char >= u - 1) and t >= min_pts and (e > 0.4 or dwell >= 10):
                x_temp[0, 2] = 1.0
                outputs.append(x_temp)
                break

            outputs.append(x_temp)
            x = x_temp.unsqueeze(0)

    deltas = (torch.cat(outputs, dim=0).cpu() * model.sd + model.mu).numpy()
    N = len(deltas)
    raw_offsets = np.zeros((N, 4), dtype=np.float32)
    raw_offsets[:, 0] = deltas[:, 0]
    raw_offsets[:, 1] = -deltas[:, 1]
    raw_offsets[:, 2] = deltas[:, 2]
    raw_offsets[:, 3] = 0.02

    scaled_offsets, _ = normalize_scale(raw_offsets, target_median=1.0)
    coords = offsets_to_coords(scaled_offsets)
    coords = denoise(coords, window=7, polyorder=3)

    base = float(np.percentile(coords[:, 1], 20))
    coords[:, 1] -= base
    coords[:, :2] *= 1.7
    return coords


def main():
    model_path = REPO_ROOT / "Handrwitten" / "checkpoints" / "pretrained"
    model = PretrainedGravesSynthesis(model_path, bias=1.0)

    test_cases = [
        ("user_case_H", "hello world HHHHHHHHH"),
        ("user_case_slash", "/////"),
        ("case_barcode_l", "llllllll"),
        ("case_gla", "GLA University - 2026"),
    ]

    out_dir = REPO_ROOT / "outputs" / "adaptive_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    style = InkStyle(color="#0f172a", width=1.6, bezier=True)

    print("Synthesizing test cases with Adaptive Controller...")
    results = {}
    for name, text in test_cases:
        coords = generate_adaptive(model, text, seed=42)
        svg = render_line(coords, style=style)
        svg_file = out_dir / f"{name}.svg"
        svg_file.write_text(svg, encoding="utf-8")
        results[name] = (svg_file, len(coords))
        print(f"  {name:20s}: {len(coords)} points -> {svg_file.name}")

    print("Rendering PNGs via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 300})
        for name, (svg_file, pts) in results.items():
            page.goto(f"file:///{svg_file.resolve().as_posix()}")
            png_file = out_dir / f"{name}.png"
            page.screenshot(path=str(png_file))
            print(f"  Captured: {png_file.name}")
        browser.close()
    print("Done! All images rendered.")


if __name__ == "__main__":
    main()
