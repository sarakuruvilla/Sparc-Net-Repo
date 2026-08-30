import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 11); ax.set_ylim(0, 7.5)
ax.axis('off')

def box(x, y, w, h, text, fc='#EAF2FB', ec='#2E5C8A', fontsize=9.5, weight='normal'):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fontsize,
             weight=weight, wrap=True)
    return (x, y, w, h)

def arrow(p1, p2, text=None, color='#333333'):
    x1 = p1[0]+p1[2]/2; y1=p1[1]
    x2 = p2[0]+p2[2]/2; y2=p2[1]+p2[3]
    a = FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='-|>', mutation_scale=14,
                         linewidth=1.3, color=color)
    ax.add_patch(a)
    if text:
        ax.text((x1+x2)/2+0.15, (y1+y2)/2, text, fontsize=8, color=color, style='italic')

# --- Input row ---
b_local = box(0.4, 6.4, 2.6, 0.7, "Local sensor window\n(24h lags, hour-of-day, wind speed)", fc='#FDF3E7', ec='#B0722A')
b_neigh = box(3.4, 6.4, 2.6, 0.7, "Neighbor features\n(PM2.5, calib. confidence,\nbearing align., distance, WS)", fc='#FDF3E7', ec='#B0722A')
b_target = box(7.2, 6.4, 3.0, 0.7, "Local target series  y_t", fc='#FDF3E7', ec='#B0722A')

# --- Series-parallel split ---
b_arfima = box(7.7, 5.0, 2.5, 0.8, "Linear stage\nARFIMA (GPH d\u0302, causal AR)\n\u2192 L\u0302_t", fc='#E9F7EF', ec='#2E7D4F')
arrow(b_target, b_arfima)

# --- CNN encoder ---
b_cnn = box(0.4, 5.0, 2.6, 0.8, "Local encoder\n3\u00d7 Conv1D(32, k=3, ReLU)\n+ Dropout", fc='#EAF2FB', ec='#2E5C8A')
arrow(b_local, b_cnn)

# --- Spatial attention ---
b_att = box(3.4, 5.0, 2.6, 0.8, "Spatial attention\ng(cos\u0394WD, L_i, WS) + log(c_i+\u03b5)\n\u2192 softmax \u03b1_i \u2192 context vector", fc='#EAF2FB', ec='#2E5C8A')
arrow(b_neigh, b_att)

# --- concat ---
b_concat = box(1.6, 3.7, 3.6, 0.7, "Channel-wise concatenation\n(per timestep)", fc='#F5EAFB', ec='#7A3FA0')
arrow(b_cnn, b_concat)
arrow(b_att, b_concat)

# --- BiGRU ---
b_bigru = box(1.6, 2.4, 3.6, 0.7, "2\u00d7 Bidirectional GRU (16 units/dir)", fc='#EAF2FB', ec='#2E5C8A')
arrow(b_concat, b_bigru)

# --- residual output ---
b_resid = box(1.6, 1.1, 3.6, 0.7, "Dense(1) \u2192 nonlinear residual  N\u0302_t", fc='#E9F7EF', ec='#2E7D4F')
arrow(b_bigru, b_resid)

# --- sum ---
b_sum = box(4.0, 0.0, 3.0, 0.7, "\u0177_t = L\u0302_t + N\u0302_t   (additive combination)", fc='#FBEAEA', ec='#B03A2E', weight='bold')
arrow(b_resid, b_sum)
arrow(b_arfima, b_sum)

ax.text(0.4, 7.25, "SPARC-Net architecture", fontsize=14, weight='bold')

plt.tight_layout()
import os
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'figures'), exist_ok=True)
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figures', 'architecture_diagram.png'), dpi=200, bbox_inches='tight')
print("saved")
