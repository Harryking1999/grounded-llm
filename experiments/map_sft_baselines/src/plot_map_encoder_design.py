from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(12,5.5))
def box(ax,x,y,w,h,text,color='#F1F5F9',edge='#94A3B8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.015',facecolor=color,edgecolor=edge,linewidth=1.1))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='#173047')
def arrow(ax,a,b,label=None,color='#6B7D8D',connection='arc3'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,linewidth=1.4,color=color,connectionstyle=connection))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2,label,color=color,fontsize=10)
for ax,mode,title in zip(axes,['address','content'],['B／E：地址生成K，地图特征生成V','C／F：同一地图特征生成K和V']):
 ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');ax.set_title(title,loc='left',fontsize=14,weight='bold',pad=12)
 box(ax,.02,.78,.24,.13,'Q向量\n＋角色／候选ID')
 box(ax,.36,.78,.25,.13,'4层\nself-attention\n＋MLP')
 box(ax,.73,.78,.25,.13,'新地图特征\nH[1] … H[L]')
 arrow(ax,(.27,.845),(.35,.845));arrow(ax,(.62,.845),(.72,.845))
 box(ax,.37,.42,.25,.13,'K = W_K a[i]' if mode=='address' else 'K = W_K h[i]','#E7F0FA','#4A80B8')
 box(ax,.73,.42,.25,.13,'V = W_V h[i]','#E6F2EE','#438E79')
 arrow(ax,(.855,.77),(.855,.56),color='#438E79')
 if mode=='address':
  box(ax,.02,.42,.25,.13,'地址 a[i]\n角色＋候选ID','#E7F0FA','#4A80B8')
  arrow(ax,(.28,.485),(.36,.485),color='#4A80B8')
  ax.text(.02,.64,'K的这条支路绕过地图编码器',fontsize=10,color='#356C9F')
 else:
  arrow(ax,(.79,.77),(.50,.56),color='#4A80B8')
  ax.text(.02,.47,'K、V各有自己的线性投影',fontsize=10,color='#526678')
 box(ax,.40,.10,.39,.16,'cross-attention\n读取地图')
 arrow(ax,(.50,.41),(.52,.27),color='#4A80B8');arrow(ax,(.855,.41),(.72,.27),color='#438E79')
 ax.text(.02,.18,'LLM隐藏状态\n生成query',fontsize=11,va='center',color='#173047')
 arrow(ax,(.29,.18),(.39,.18))
fig.text(.03,.05,'两组均保留 L = 2＋合法动作数 个槽位；前4层self-attention内部均使用普通K/V。',fontsize=12,color='#475569')
fig.subplots_adjust(left=.03,right=.985,bottom=.14,top=.88,wspace=.16)
out=Path('experiments/map_sft_baselines/results/figures')
for ext in ['png','pdf']:fig.savefig(out/f'map_encoder_kv_sources.{ext}',dpi=180,facecolor='white')
print('rendered K/V source diagram')
