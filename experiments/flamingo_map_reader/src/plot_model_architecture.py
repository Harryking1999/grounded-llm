"""Model structure: frozen LM and map, shared memory, independent layer interfaces.

Schematic-led composite. Python is the existing reporting backend. The diagram
uses the same geometry for editable SVG and the raster review/export. Colors
encode parameter status; no quantitative or causal claim is implied.
"""
from pathlib import Path
from html import escape
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1600, 1170
B, BF = "#376E9B", "#EAF2F8"
O, OF = "#BD713E", "#FFF0E4"
INK, GRAY, LINE = "#263643", "#64727C", "#81909B"
plt.rcParams.update({"font.family": "Microsoft YaHei", "svg.fonttype": "none"})
fig = plt.figure(figsize=(16,11.7), facecolor="white")
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="1600" height="1170" fill="white"/>']

def rect(x,y,w,h,fill="white",stroke=LINE,r=12,lw=1.6,dash=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill,edgecolor=stroke,linewidth=lw*.72,linestyle="--" if dash else "-"))
    da=' stroke-dasharray="8 6"' if dash else ''
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"{da}/>')

def txt(x,y,s,size=22,color=INK,weight="normal",align="center"):
    ax.text(x,y,s,fontsize=size*.72,color=color,ha=align,va="center",weight=weight)
    anch={"center":"middle","left":"start","right":"end"}[align]
    svg.append(f'<text x="{x}" y="{y+size*.34}" text-anchor="{anch}" font-family="Microsoft YaHei, Noto Sans SC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(s)}</text>')

def arrow(points,color=LINE,width=2.0,dash=False,head=True):
    xx,yy=zip(*points)
    ax.plot(xx,yy,color=color,lw=width*.72,linestyle="--" if dash else "-",solid_capstyle="round")
    ss=" ".join(f"{x},{y}" for x,y in points)
    da=' stroke-dasharray="7 6"' if dash else ''
    svg.append(f'<polyline points="{ss}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{da}/>')
    if head:
        x,y=points[-1]; px,py=points[-2]
        if x==px:
            sign=1 if y>py else -1; tri=[(x,y),(x-5,y-sign*10),(x+5,y-sign*10)]
        else:
            sign=1 if x>px else -1; tri=[(x,y),(x-sign*10,y-5),(x-sign*10,y+5)]
        ax.add_patch(Polygon(tri,color=color))
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in tri)+f'" fill="{color}"/>')

def circle(x,y,r,fill="white",stroke=LINE):
    ax.add_patch(Circle((x,y),r,facecolor=fill,edgecolor=stroke,lw=1.5))
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

def box(x,y,w,h,label,train=False,sub=None):
    rect(x,y,w,h,OF if train else BF,O if train else B)
    txt(x+w/2,y+h/2-(12 if sub else 0),label,22,weight="bold")
    if sub: txt(x+w/2,y+h/2+18,sub,18,GRAY)

txt(55,48,"模型架构：在冻结语言模型中逐层加入地图读取接口",31,weight="bold",align="left")
txt(55,91,"共享地图 memory；每层独立投影与门控。蓝色模块冻结，橙色模块参与 SFT。",21,GRAY,align="left")
rect(1302,82,18,18,BF,B,r=3); txt(1330,92,"冻结",19,B,align="left")
rect(1417,82,18,18,OF,O,r=3); txt(1445,92,"可训练",19,O,align="left")

# A. A real layer stack, separate from the environment interaction flow.
txt(60,144,"A  语言模型层堆叠",24,weight="bold",align="left")
txt(255,193,"任务描述、历史与候选编号",20,GRAY)
arrow([(255,211),(255,236)])
box(75,236,360,66,"Qwen2.5-1.5B 文本 Embedding")
arrow([(255,302),(255,343)])
rect(57,343,396,504,"#FBFCFD","#CAD3DA",r=18)
txt(255,374,"每个 Decoder 前插入读取接口",20,GRAY)
box(84,407,342,67,"Gated Cross-Attention 1",True)
arrow([(255,474),(255,494)])
box(84,494,342,80,"原 Decoder 1",False,"Self-Attention ＋ FFN")
arrow([(255,574),(255,604)])
for yy in (614,625,636):
    circle(255,yy,2.2,GRAY,GRAY)
arrow([(255,647),(255,671)])
box(84,671,342,67,"Gated Cross-Attention L",True)
arrow([(255,738),(255,758)])
box(84,758,342,66,"原 Decoder L",False,"Self-Attention ＋ FFN")
arrow([(255,824),(255,893)])
box(75,893,360,66,"原模型末端 Norm ＋ LM Head")
arrow([(255,959),(255,994)])
rect(75,994,360,74,"white","#A8B4BE")
txt(255,1016,"自回归生成文本",21,weight="bold")
txt(255,1044,"候选排序 ／ 动作 ／ 终止判断",20)

# B. Shared memory. The address and value branches are intentionally separate.
rect(496,135,1045,308,"#FBFCFD","#CAD3DA",r=18)
txt(521,166,"B  共享地图 memory：地址与状态内容分开编码",24,weight="bold",align="left")
txt(781,217,"角色：当前 / 目标 / 后继    编号：0 / 1…k",19,GRAY)
box(558,245,438,72,"角色 / 编号 Embedding",True,"各 32 维，拼接为 64 维地址")
txt(778,358,"地址 memory",21,O,weight="bold")
arrow([(778,317),(778,341)],O)

txt(1269,215,"冻结地图输出的 128 维状态向量",20,B)
for x,lab in [(1083,"当前"),(1206,"目标"),(1330,"后继 1…k")]:
    rect(x,240,111,44,BF,B,r=5); txt(x+55.5,261,lab,18,B)
arrow([(1138,284),(1138,294),(1385,294),(1385,284)],B,head=False)
arrow([(1268,284),(1268,305)],B)
box(1083,305,371,58,"线性投影 P：128 → 256",True)
txt(1268,395,"状态内容 memory：P(z)",21,O,weight="bold")
arrow([(1268,363),(1268,378)],O)

# C. The inserted module; projection weights here are distinct at each layer.
rect(496,485,1045,596,"#FFFCF9","#D9C7B9",r=18)
txt(521,516,"C  第 ℓ 层读取接口（放大）",24,weight="bold",align="left")
arrow([(778,374),(778,461),(976,461),(976,574)],O)
arrow([(1268,411),(1268,574)],O)
txt(831,444,"送至每层",17,O,align="left")
box(870,574,212,61,"LayerNorm → Wk",True)
box(1162,574,212,61,"Wv",True)
txt(976,662,"K",23,O,weight="bold")
txt(1268,662,"V",23,O,weight="bold")

txt(599,580,"语言隐藏状态 hℓ",20,B,weight="bold")
arrow([(598,602),(598,676)],B)
box(516,676,219,61,"LayerNorm → Wq",True)
arrow([(735,706),(840,706)],O)
txt(783,684,"Query",19,O)
box(840,680,284,79,"多头 Cross-Attention",True,"4 heads × 64 维")
arrow([(976,635),(976,680)],O)
arrow([(1268,635),(1268,719),(1124,719)],O)
txt(1361,783,"padding / 轮次 mask",18,GRAY)
arrow([(1245,783),(1207,783),(1207,766),(1095,766),(1095,759)],GRAY,1.5,True)

arrow([(982,759),(982,794)],O)
box(880,794,205,50,"输出投影 Wo",True)
arrow([(982,844),(982,870)],O)
box(852,870,260,54,"× tanh(αℓ)",True)
txt(1243,897,"可训练标量门控",19,O)
arrow([(982,924),(982,949)],O)
circle(982,970,21,"white",O); txt(982,969,"+",27,O)
arrow([(598,604),(500,604),(500,970),(961,970)],B,2)
txt(631,944,"原隐藏状态：残差直连",19,B)
arrow([(982,991),(982,1011)],B)
box(821,1011,439,52,"原 Decoder ℓ（Self-Attention ＋ FFN）")
txt(1381,1022,"冻结",19,B)
txt(1381,1050,"进入下一层",18,GRAY)
arrow([(1260,1037),(1309,1037)],B)

# Bracket makes the stack-to-detail relation explicit without crossing data edges.
arrow([(426,440),(473,440),(473,502),(496,502)],O,1.8,True)
txt(54,1120,"训练范围：角色/编号 Embedding、线性投影 P、各层 Cross-Attention 与门控；原语言模型和地图保持冻结。",20,INK,align="left")
txt(54,1152,"接口没有额外 FFN。图中共享的是地图 memory；Q/K/V/O 投影、LayerNorm 与门控均按层独立。",19,GRAY,align="left")

svg.append('</svg>')
(OUT/'progress_model_architecture.svg').write_text('\n'.join(svg),encoding='utf-8')
fig.savefig(OUT/'progress_model_architecture.png',dpi=150,facecolor='white')
plt.close(fig)
print('Saved progress_model_architecture.svg and progress_model_architecture.png')
