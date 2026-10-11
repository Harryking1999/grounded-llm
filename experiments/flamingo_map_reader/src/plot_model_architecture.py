"""Compact, editable schematic of the implemented gated map reader.

The existing Python figure workflow generates SVG and PNG from identical
geometry. Frozen map content and role/ID addresses form separate V/K branches;
the language state supplies Q. One layer is expanded; interfaces are independent
at every decoder layer and read the same map memory. No extra FFN is introduced.
"""
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon


OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
WIDTH, HEIGHT = 1120, 700
INK = "#222222"
BLUE, BLUE_LIGHT = "#9EBBDC", "#E7EFF8"
PEACH, CREAM = "#EAC4A7", "#FBF5E9"
GRAY = "#C7CCD2"
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})


class Drawing:
    """Mirror the simple editable SVG primitives in the Python raster preview."""

    def __init__(self):
        self.fig = plt.figure(figsize=(11.2, 7), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
        self.ax.axis("off")
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<rect width="1120" height="700" fill="white"/>',
        ]

    def rect(self, x, y, w, h, fill="white", stroke=INK, radius=3, width=1.2):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
            edgecolor=stroke if stroke != "none" else "none", linewidth=width*.72))
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def text(self, x, y, text, size=20, weight="normal", italic=False, anchor="middle"):
        alignment = {"middle": "center", "start": "left", "end": "right"}[anchor]
        self.ax.text(x, y, text, fontsize=size*.72, color=INK, ha=alignment,
            va="center", weight=weight, style="italic" if italic else "normal")
        style = ' font-style="italic"' if italic else ''
        self.svg.append(f'<text x="{x}" y="{y+size*.34}" text-anchor="{anchor}" '
            f'font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}"{style} fill="{INK}">{escape(text)}</text>')

    def arrow(self, points, width=1.2, head=True, dashed=False):
        xx, yy = zip(*points)
        self.ax.plot(xx, yy, color=INK, lw=width*.72,
            linestyle=(0, (4, 4)) if dashed else "-", solid_capstyle="round")
        dash = ' stroke-dasharray="4 4"' if dashed else ''
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{INK}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dash}/>')
        if head:
            x, y = points[-1]; px, py = points[-2]
            if x == px:
                direction = 1 if y > py else -1
                triangle = [(x, y), (x-3.5, y-direction*7), (x+3.5, y-direction*7)]
            else:
                direction = 1 if x > px else -1
                triangle = [(x, y), (x-direction*7, y-3.5), (x-direction*7, y+3.5)]
            self.ax.add_patch(Polygon(triangle, facecolor=INK, edgecolor="none"))
            self.svg.append('<polygon points="'+" ".join(f"{a},{b}" for a,b in triangle)+f'" fill="{INK}"/>')

    def circle(self, x, y, r=12, fill="white", width=1.2):
        self.ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=INK, lw=width*.72))
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
            f'stroke="{INK}" stroke-width="{width}"/>')

    def dots(self, x, y):
        for delta in (-8, 0, 8):
            self.circle(x, y+delta, r=1.3, fill=INK, width=.2)

    def matrix(self, x, y, color, cols=5, rows=4, cell=15):
        self.svg.append('<g>')
        for row in range(rows):
            for col in range(cols):
                self.rect(x+col*cell, y+row*cell, cell, cell,
                    color, "white", radius=0, width=1)
        self.svg.append('</g>')

    def block(self, x, y, w, h, label, fill="white", size=19, italic=False):
        self.rect(x, y, w, h, fill)
        self.text(x+w/2, y+h/2, label, size, italic=italic)

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.svg.append('</svg>')
        (OUT/'progress_model_architecture.svg').write_text("\n".join(self.svg), encoding="utf-8")
        self.fig.savefig(OUT/'progress_model_architecture.png', dpi=220, facecolor="white")
        plt.close(self.fig)


def main():
    d = Drawing()
    # The cream field expands one interface; a second layer's adapter is shown.
    d.rect(35, 165, 205, 450, width=1.7)
    d.rect(822, 63, 244, 594, width=1.7)
    d.rect(282, 157, 518, 363, CREAM, "none", radius=8)
    d.text(137, 137, "Frozen Q-map", 23)
    d.text(944, 37, "Frozen Qwen2.5-1.5B", 23)
    d.text(542, 183, "Trainable interface ℓ", 23, weight="bold")

    # One frozen map, supplying shared content memory.
    d.text(137, 668, "States", 20)
    d.arrow([(137, 649), (137, 572)])
    d.block(64, 550, 146, 22, "", BLUE_LIGHT)
    d.arrow([(137, 550), (137, 536)])
    d.dots(137, 519)
    d.arrow([(137, 503), (137, 497)])
    d.block(64, 475, 146, 22, "", BLUE_LIGHT)
    d.arrow([(137, 475), (137, 373)])
    d.matrix(77, 277, BLUE, cols=6, rows=5, cell=20)
    d.text(137, 246, "State tokens", 19)

    # Content (V) and address (K) stay separate; matrix cells are schematic.
    d.arrow([(197, 327), (260, 327), (260, 555), (328, 555)])
    d.block(328, 538, 108, 34, "P", PEACH, size=23, italic=True)
    d.arrow([(382, 538), (382, 499)])
    d.matrix(345, 439, BLUE)
    d.arrow([(382, 439), (382, 406)])
    d.block(329, 374, 106, 32, "Wv", size=21, italic=True)
    d.arrow([(382, 374), (382, 333)])
    d.text(398, 352, "V", 18, italic=True)

    d.text(542, 668, "Role / ID", 20)
    d.arrow([(542, 649), (542, 572)])
    d.block(473, 538, 138, 34, "Embedding", PEACH, size=18)
    d.text(428, 600, "Shared memory", 17)
    d.arrow([(542, 538), (542, 499)])
    d.matrix(505, 439, PEACH)
    d.arrow([(542, 439), (542, 406)])
    d.block(483, 374, 118, 32, "LN → Wk", size=19)
    d.arrow([(542, 374), (542, 333)])
    d.text(558, 352, "K", 18, italic=True)

    # h supplies Q and the unchanged residual path.
    d.matrix(680, 439, GRAY)
    d.arrow([(717, 439), (717, 406)])
    d.block(657, 374, 120, 32, "LN → Wq", size=19)
    d.arrow([(717, 374), (717, 333)])
    d.text(733, 352, "Q", 18, italic=True)
    d.arrow([(944, 492), (944, 249)])
    d.arrow([(944, 469), (755, 469)])
    d.text(900, 452, "hℓ", 20, italic=True)

    # No interface FFN: attention -> Wo -> scalar gate -> residual.
    d.block(317, 293, 456, 40, "Cross-attention", size=22)
    d.arrow([(542, 293), (542, 254)])
    d.block(482, 220, 120, 34, "Wo", size=21, italic=True)
    d.arrow([(602, 237), (651, 237)])
    d.block(651, 220, 122, 34, "× tanh(αℓ)", size=19)
    d.arrow([(773, 237), (932, 237)])
    d.circle(944, 237)
    d.text(944, 237, "+", 23)
    d.arrow([(944, 225), (944, 198)])

    # Two distinct adapter layers are depicted in the frozen LM stack.
    d.text(944, 689, "Text tokens", 20)
    d.arrow([(944, 674), (944, 651)])
    d.block(850, 617, 188, 34, "Embedding", BLUE_LIGHT)
    d.dots(944, 601)
    d.arrow([(944, 589), (944, 588)])
    d.block(850, 554, 188, 34, "Adapter ℓ−1", PEACH, size=19)
    d.arrow([(944, 554), (944, 530)])
    d.block(850, 492, 188, 38, "Decoder ℓ−1", BLUE_LIGHT, size=19)
    d.block(850, 155, 188, 43, "Decoder ℓ", BLUE_LIGHT, size=20)
    d.arrow([(944, 155), (944, 148)])
    d.dots(944, 134)
    d.arrow([(944, 120), (944, 112)])
    d.block(850, 78, 188, 34, "LM head", BLUE_LIGHT, size=19)

    d.rect(318, 633, 14, 14, BLUE_LIGHT, width=.8)
    d.text(342, 641, "Frozen", 16, anchor="start")
    d.rect(669, 633, 14, 14, PEACH, width=.8)
    d.text(693, 641, "Trainable", 16, anchor="start")
    d.save()
    print("Saved progress_model_architecture.svg and progress_model_architecture.png")


if __name__ == "__main__":
    main()
