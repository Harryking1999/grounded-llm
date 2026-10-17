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
WIDTH, HEIGHT = 1320, 860
INK = "#222222"
BLUE, BLUE_LIGHT = "#9EBBDC", "#E7EFF8"
PEACH, CREAM = "#EAC4A7", "#FBF5E9"
GRAY = "#C7CCD2"
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})


class Drawing:
    """Mirror the simple editable SVG primitives in the Python raster preview."""

    def __init__(self):
        self.fig = plt.figure(figsize=(WIDTH/100, HEIGHT/100), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
        self.ax.axis("off")
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
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

    def arrow(self, points, width=1.2, head=True, dashed=False, color=INK):
        xx, yy = zip(*points)
        self.ax.plot(xx, yy, color=color, lw=width*.72,
            linestyle=(0, (4, 4)) if dashed else "-", solid_capstyle="round")
        dash = ' stroke-dasharray="4 4"' if dashed else ''
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
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
    # Only the selected complete layer is expanded in the right stack. The
    # middle view magnifies its orange component, not another processing path.
    d.rect(1015, 63, 270, 753, width=1.6)
    d.text(1150, 37, "Qwen2.5-1.5B", 24)
    d.block(1050, 78, 200, 32, "LM head", BLUE_LIGHT)

    def layer(y, label, selected=False):
        d.rect(1030, y, 240, 164, "#FFFCF8" if selected else "white",
            "#B88B66" if selected else "#A7A7A7", radius=5,
            width=1.8 if selected else 1.1)
        d.text(1043, y+23, label, 20, weight="bold" if selected else "normal", anchor="start")
        d.block(1050, y+52, 200, 34, "Frozen Decoder", BLUE_LIGHT, size=18)
        d.block(1050, y+110, 200, 34, "Gated Cross-Attn", PEACH, size=18)
        d.arrow([(1150, y+110), (1150, y+86)])

    d.block(1050, 225, 200, 34, "Layer L", "#EEF0F2")
    layer(335, "Layer ℓ", selected=True)
    d.block(1050, 625, 200, 34, "Layer 1", "#EEF0F2")
    d.arrow([(1150, 225), (1150, 110)])
    d.arrow([(1150, 387), (1150, 327)])
    d.dots(1150, 313)
    d.arrow([(1150, 300), (1150, 259)])
    d.arrow([(1150, 625), (1150, 542)])
    d.dots(1150, 528)
    d.arrow([(1150, 515), (1150, 479)])
    d.block(1050, 760, 200, 34, "Embedding", BLUE_LIGHT)
    d.arrow([(1150, 760), (1150, 659)])
    d.text(1150, 841, "Text tokens", 20)
    d.arrow([(1150, 825), (1150, 794)])

    # Dashed, arrowless magnification links carry no computation.
    d.arrow([(1050, 445), (949, 166)], head=False, dashed=True, color="#A3A3A3")
    d.arrow([(1050, 479), (949, 656)], head=False, dashed=True, color="#A3A3A3")
    d.rect(292, 166, 657, 490, CREAM, "#C6AF91", radius=7, width=1.3)
    d.text(595, 193, "Gated Cross-Attn ℓ", 23, weight="bold")
    d.text(595, 218, "Expanded view", 16)

    # One frozen state map, shared by the layer-specific interfaces.
    d.rect(35, 295, 195, 350, width=1.6)
    d.text(132, 268, "Frozen Q-map", 23)
    d.text(132, 696, "States", 20)
    d.arrow([(132, 678), (132, 605)])
    d.block(64, 583, 137, 22, "", BLUE_LIGHT)
    d.arrow([(132, 583), (132, 573)])
    d.dots(132, 560)
    d.arrow([(132, 546), (132, 540)])
    d.block(64, 518, 137, 22, "", BLUE_LIGHT)
    d.arrow([(132, 518), (132, 445)])
    d.matrix(72, 365, BLUE, cols=6, rows=4, cell=20)
    d.text(132, 341, "State tokens", 19)

    # Shared P and role/ID embeddings lie outside the per-layer expanded box.
    d.arrow([(192, 405), (260, 405), (260, 697), (330, 697)])
    d.block(330, 680, 100, 34, "P", PEACH, size=23, italic=True)
    d.arrow([(380, 680), (380, 552)])
    d.matrix(342.5, 492, BLUE)
    d.arrow([(380, 492), (380, 459)])
    d.block(327, 427, 106, 32, "Wv", size=21, italic=True)
    d.arrow([(380, 427), (380, 380)])
    d.text(396, 403, "V", 18, italic=True)

    d.text(550, 788, "Role / ID", 20)
    d.arrow([(550, 771), (550, 714)])
    d.block(480, 680, 140, 34, "Embedding", PEACH, size=18)
    d.text(425, 739, "Shared memory", 17)
    d.arrow([(550, 680), (550, 552)])
    d.matrix(512.5, 492, PEACH)
    d.arrow([(550, 492), (550, 459)])
    d.block(491, 427, 118, 32, "LN → Wk", size=19)
    d.arrow([(550, 427), (550, 380)])
    d.text(566, 403, "K", 18, italic=True)

    # Q, attention output, gate and residual are all inside this same adapter.
    d.text(752, 621, "hℓ", 20, italic=True)
    d.arrow([(752, 605), (752, 552)])
    d.matrix(714.5, 492, GRAY)
    d.arrow([(752, 492), (752, 459)])
    d.block(692, 427, 120, 32, "LN → Wq", size=19)
    d.arrow([(752, 427), (752, 380)])
    d.text(768, 403, "Q", 18, italic=True)
    d.arrow([(752, 583), (865, 583), (865, 287)])
    d.block(327, 340, 485, 40, "Cross-attention", size=22)
    d.arrow([(567, 340), (567, 292)])
    d.block(517, 258, 100, 34, "Wo", size=21, italic=True)
    d.arrow([(617, 275), (672, 275)])
    d.block(672, 258, 135, 34, "× tanh(αℓ)", size=19)
    d.arrow([(807, 275), (853, 275)])
    d.circle(865, 275)
    d.text(865, 275, "+", 23)
    d.arrow([(865, 263), (865, 234)])
    d.text(895, 237, "h′ℓ", 20, italic=True)

    d.rect(315, 817, 14, 14, BLUE_LIGHT, width=.8)
    d.text(339, 825, "Frozen", 16, anchor="start")
    d.rect(702, 817, 14, 14, PEACH, width=.8)
    d.text(726, 825, "Trainable", 16, anchor="start")
    d.save()
    print("Saved progress_model_architecture.svg and progress_model_architecture.png")


if __name__ == "__main__":
    main()
