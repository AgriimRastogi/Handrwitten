import json
import torch
import sys
import asyncio
from pathlib import Path

# Add Handrwitten to sys.path
sys.path.insert(0, str(Path.cwd() / "Handrwitten"))
import numpy as np

from hws.pretrained_model import IAMTokenizer, SynthesisNetwork
from hws.render import render_page, PageLayout, InkStyle
from playwright.async_api import async_playwright

model_dir = Path("Handrwitten/checkpoints/pretrained")
with open(model_dir / "meta.json") as f:
    meta = json.load(f)

tokenizer = IAMTokenizer(meta["charset"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from hws.pretrained_model import PretrainedGravesSynthesis
synth = PretrainedGravesSynthesis(model_dir, device=device)

examples = [
    {
        "filename": "example_lab_notes",
        "title": "Example 1: University Capstone Lab Notes",
        "lines": [
            "Stroke Level Handwriting Synthesis",
            "Deep Recurrent Neural Networks",
            "Vector Generative Modeling",
            "GLA University Greater Noida",
        ],
        "ruled": True,
        "margin_rule": True,
        "color": "#16213e", # Midnight Navy
        "width": 1.7,
        "line_height": 46.0,
    },
    {
        "filename": "example_quote",
        "title": "Example 2: Classic Calligraphic Pangrams",
        "lines": [
            "The quick brown fox jumps",
            "over the lazy dog.",
            "Sphinx of black quartz,",
            "judge my vow.",
        ],
        "ruled": True,
        "margin_rule": True,
        "color": "#0f3460", # Royal Blue
        "width": 1.6,
        "line_height": 44.0,
    },
    {
        "filename": "example_diary_memo",
        "title": "Example 3: Project Meeting Memo",
        "lines": [
            "Meeting Notes - 27 September 2026",
            "Model verified on IAM benchmark",
            "Real-time stroke vector synthesis",
            "Status: Ahead of schedule",
        ],
        "ruled": True,
        "margin_rule": True,
        "color": "#2c1810", # Warm Sepia / Fountain Pen Brown
        "width": 1.65,
        "line_height": 45.0,
    },
]

print("Generating showcase examples...")

for ex in examples:
    rendered_lines = []
    print(f"\n--- {ex['title']} ---")
    for idx, line_text in enumerate(ex["lines"], start=1):
        pts = synth.generate(line_text, temp=0.35)
        rendered_lines.append(pts)
        print(f"  Line {idx}: '{line_text}' -> {len(pts)} points")
        
    num_lines = len(rendered_lines)
    layout = PageLayout(
        ruled=ex["ruled"],
        margin_rule=ex["margin_rule"],
        line_height=ex["line_height"],
        margin_top=75.0,
        margin_left=70.0,
        height=max(500.0, 100.0 + num_lines * ex["line_height"] + 150.0),
    )
    style = InkStyle(
        color=ex["color"],
        width=ex["width"],
        bezier=True,
    )
    
    svg = render_page(rendered_lines, layout=layout, style=style, seed=42)
    svg_path = Path("outputs") / f"{ex['filename']}.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"Saved: {svg_path}")

async def snap_all():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for ex in examples:
            svg_path = Path("outputs") / f"{ex['filename']}.svg"
            png_path = Path("outputs") / f"{ex['filename']}.png"
            page = await b.new_page(viewport={"width": 650, "height": 480})
            await page.goto(f"file:///{svg_path.resolve().as_posix()}")
            await page.screenshot(path=str(png_path))
            await page.close()
            print(f"Captured: {png_path}")
        await b.close()

asyncio.run(snap_all())
print("\nAll showcase examples successfully generated and screenshotted!")
