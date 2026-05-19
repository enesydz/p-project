# ╔══════════════════════════════════════════════════════════════════╗
# ║  PART 2 — CHARTLAR · VİZÜALİZASYON · PROFESYONEL NETWORK       ║
# ║  PART 1 çalıştırıldıktan sonra bu hücreyi çalıştırın            ║
# ╚══════════════════════════════════════════════════════════════════╝

import math as _math
import plotly.graph_objects as go
import os
from IPython.display import HTML, display

_N  = len(SEGMENT_SIRASI)
_ND = len(DONEM_SIRASI)
xs  = np.arange(_N)

# ══════════════════════════════════════════════════════════════════
# § A — GENEL BAKIŞ  (6 panel)
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 11))
gs  = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.40)
ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1]); ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,0]); ax5 = fig.add_subplot(gs[1,1]); ax6 = fig.add_subplot(gs[1,2])

b1 = ax1.bar(xs-0.2, PENETRASYON["Alim_Pct"],  width=0.38, color=SEG_RENK_LIST, alpha=0.9,  zorder=3)
b2 = ax1.bar(xs+0.2, PENETRASYON["Satim_Pct"], width=0.38, color=SEG_RENK_LIST, alpha=0.45, zorder=3, hatch="//")
for bg in [b1,b2]:
    for p in bg.patches:
        h = p.get_height()
        if h > 1: ax1.text(p.get_x()+p.get_width()/2, h+1, f"{h:.0f}%",
                            ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=TEXT_CLR)
ax1.set_xticks(xs); ax1.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax1.set_ylabel("Penetrasyon (%)"); ax1.set_ylim(0, 115)
ax1.set_title("Fon Penetrasyonu\n(Alım / Satım Yapan Müşteri %)")
ax1.legend(handles=[mpatches.Patch(facecolor="#555",label="Alım"),
                    mpatches.Patch(facecolor="#aaa",hatch="//",label="Satım")],
           loc="upper right", fontsize=8); style_axes(ax1)

aum_m = AUM / 1e6
b3 = ax2.bar(xs, aum_m.values, color=SEG_RENK_LIST, alpha=0.9, zorder=3, edgecolor="white", linewidth=0.8)
for p in b3.patches:
    h = p.get_height()
    if h > 0:
        lbl = f"{h:.0f}M" if h < 1000 else f"{h/1e3:.1f}B"
        ax2.text(p.get_x()+p.get_width()/2, h+aum_m.max()*0.015,
                 lbl, ha="center", va="bottom", fontsize=7, fontweight="bold", color=TEXT_CLR)
ax2.set_xticks(xs); ax2.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax2.set_ylabel("Net AUM (M TL)"); ax2.set_title("Net Fon Pozisyonu\n(Toplam Alım − Satım)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M")); style_axes(ax2)

ort_alim  = METRIK_GENEL[METRIK_GENEL["event_type"]=="Alım"].set_index("musteri_segmenti")["Ort_Tutar_K"].reindex(SEGMENT_SIRASI)
ort_satim = METRIK_GENEL[METRIK_GENEL["event_type"]=="Satım"].set_index("musteri_segmenti")["Ort_Tutar_K"].reindex(SEGMENT_SIRASI)
y_pos = np.arange(_N)
ax3.scatter(ort_alim.values,  y_pos, s=150, c=SEG_RENK_LIST, marker="o", zorder=4, label="Alım ort.", linewidths=1.5, edgecolors="white")
ax3.scatter(ort_satim.values, y_pos, s=150, c=SEG_RENK_LIST, marker="D", zorder=4, label="Satım ort.", alpha=0.65, linewidths=1.5, edgecolors="white")
for i, seg in enumerate(SEGMENT_SIRASI):
    a, s_ = ort_alim.iloc[i], ort_satim.iloc[i]
    ax3.plot([min(a,s_),max(a,s_)],[i,i], color=SEG_RENK[seg], lw=1.5, alpha=0.4, zorder=2)
ax3.set_yticks(y_pos); ax3.set_yticklabels(SEG_SHORT, fontsize=7.5)
ax3.set_xlabel("Ortalama İşlem Tutarı (Bin TL)"); ax3.set_title("Ort. İşlem Tutarı\n(● Alım  ◆ Satım)")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}K"))
ax3.legend(loc="lower right", fontsize=8); style_axes(ax3)

ax4.barh(xs+0.2, SADAKAT["Cok_Donem_Pct"].values, height=0.35, color=SEG_RENK_LIST, alpha=0.9, label="Her 2 dönemde", zorder=3)
ax4.barh(xs-0.2, SADAKAT["Tek_Donem_Pct"].values,  height=0.35, color=SEG_RENK_LIST, alpha=0.40, label="Tek dönem", zorder=3)
ax4.set_yticks(xs); ax4.set_yticklabels(SEG_SHORT, fontsize=7.5)
ax4.set_xlabel("Müşteri Oranı (%)"); ax4.set_title("Müşteri Sadakati\n(Her İki Dönem vs Tek Dönem Alım)")
ax4.axvline(50, ls="--", color=SPINE_CLR, lw=1.0, alpha=0.7)
ax4.legend(loc="lower right", fontsize=8); style_axes(ax4)

bd_pct = (musteri_bakiye_dilim.merge(musteri_df, on="musteri_id")
          .groupby(["musteri_segmenti","bakiye_dilimi"], observed=True).size()
          .unstack(fill_value=0).reindex(SEGMENT_SIRASI)
          .reindex(columns=BAKIYE_DILIMLERI, fill_value=0))
bd_pct_norm = bd_pct.div(bd_pct.sum(axis=1), axis=0) * 100
bottom = np.zeros(_N)
for dil in BAKIYE_DILIMLERI:
    vals = bd_pct_norm[dil].values
    ax5.bar(xs, vals, bottom=bottom, width=0.55, label=dil,
            color=BAKIYE_RENK[dil], alpha=0.9, zorder=3, edgecolor="white", linewidth=0.6)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 8: ax5.text(xs[i], b+v/2, f"{v:.0f}%", ha="center", va="center",
                            fontsize=7, fontweight="bold", color=TEXT_CLR)
    bottom += vals
ax5.set_xticks(xs); ax5.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax5.set_ylabel("Müşteri Oranı (%)"); ax5.set_ylim(0, 115)
ax5.set_title("Bakiye Dilimi Dağılımı\n(Segment içi percentile)")
ax5.legend(loc="upper right", fontsize=8, title="Dilim"); style_axes(ax5)

t_alim_m  = (alim_df.merge(musteri_df,  on="musteri_id")
             .groupby("musteri_segmenti", observed=True)["alim_tutari"].sum()
             .reindex(SEGMENT_SIRASI) / 1e6)
t_satim_m = (satim_df.merge(musteri_df, on="musteri_id")
             .groupby("musteri_segmenti", observed=True)["satim_tutari"].sum()
             .reindex(SEGMENT_SIRASI) / 1e6)
ax6.bar(xs-0.2, t_alim_m.values,  width=0.38, color=SEG_RENK_LIST, alpha=0.9,  zorder=3)
ax6.bar(xs+0.2, t_satim_m.values, width=0.38, color=SEG_RENK_LIST, alpha=0.45, zorder=3, hatch="//")
ax6.set_xticks(xs); ax6.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax6.set_ylabel("Toplam Hacim (M TL)"); ax6.set_title("Fon İşlem Hacmi\n(Alım vs Satım, M TL)")
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M"))
ax6.legend(handles=[mpatches.Patch(facecolor="#555",label="Alım"),
                    mpatches.Patch(facecolor="#aaa",hatch="//",label="Satım")],
           loc="upper left", fontsize=8); style_axes(ax6)

style_fig(fig, "A — Genel Bakış · Banka Fon Müşteri Analizi  ·  2025.09 & 2026.03",
          f"Penetrasyon · AUM · Sadakat · Bakiye Dilimi  |  {len(events_df):,} olay · {_N} segment · 2 dönem")
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()


# ══════════════════════════════════════════════════════════════════
# § B — FON ALIM DAVRANIŞI  (6 panel)
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.40)
ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1]); ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,0]); ax5 = fig.add_subplot(gs[1,1]); ax6 = fig.add_subplot(gs[1,2])

for urun in URUNLER:
    bottom = np.zeros(_N) if urun == URUNLER[0] else bottom
    vals = URUN_PRE_BUY[urun].values
    ax1.bar(xs, vals, bottom=bottom, width=0.55, label=urun,
            color=URUN_RENKLER[urun], alpha=0.9, zorder=3, edgecolor="white", linewidth=0.5)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 9: ax1.text(xs[i], b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=6.5, fontweight="bold", color="white")
    bottom += vals
ax1.set_xticks(xs); ax1.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax1.set_ylabel("İşlem Dağılımı (%)"); ax1.set_ylim(0,115)
ax1.set_title(f"Pre-Buy Ürün Mix\n(Alım öncesi −{X_DAYS} gün)")
ax1.legend(loc="upper right", fontsize=8, title="Ürün Grubu"); style_axes(ax1)

bottom = np.zeros(_N)
for urun in URUNLER:
    vals = URUN_POST_BUY[urun].values
    ax2.bar(xs, vals, bottom=bottom, width=0.55, label=urun,
            color=URUN_RENKLER[urun], alpha=0.9, zorder=3, edgecolor="white", linewidth=0.5)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 9: ax2.text(xs[i], b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=6.5, fontweight="bold", color="white")
    bottom += vals
ax2.set_xticks(xs); ax2.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax2.set_ylabel("İşlem Dağılımı (%)"); ax2.set_ylim(0,115)
ax2.set_title(f"Post-Buy Ürün Mix\n(Alım sonrası +{X_DAYS} gün)")
ax2.legend(loc="upper right", fontsize=8, title="Ürün Grubu"); style_axes(ax2)

w = 0.35
ax3.bar(xs-w/2, GIRIS_PRE_BUY.values,  width=w, color=YON_RENK["Giriş"], alpha=0.9,  label="Pre-Buy",  zorder=3)
ax3.bar(xs+w/2, GIRIS_POST_BUY.values, width=w, color=YON_RENK["Giriş"], alpha=0.50, label="Post-Buy", zorder=3)
ax3.axhline(50, ls="--", color=SPINE_CLR, lw=1.0, alpha=0.7)
for i, (a, b_) in enumerate(zip(GIRIS_PRE_BUY.values, GIRIS_POST_BUY.values)):
    ax3.text(i-w/2, a+1, f"{a:.0f}%",  ha="center", va="bottom", fontsize=7, fontweight="bold", color=TEXT_CLR)
    ax3.text(i+w/2, b_+1, f"{b_:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold", color=TEXT_CLR)
ax3.set_xticks(xs); ax3.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax3.set_ylabel("Giriş Oranı (%)"); ax3.set_ylim(0, 85)
ax3.set_title("Para Giriş Oranı\n(Pre-Buy vs Post-Buy)")
ax3.legend(fontsize=8); style_axes(ax3)

days = GUNLUK_ALIM.index.values
ax4.bar(days, GUNLUK_ALIM.values/1e3,
        color=[YON_RENK["Giriş"] if d < 0 else "#1D4ED8" for d in days],
        alpha=0.85, zorder=3, width=0.8)
ax4.axvline(0, ls="--", color="#F59E0B", lw=2.0, zorder=5)
ax4.set_xlabel("Alım Olayına Göre Gün"); ax4.set_ylabel("Ort. İşlem Tutarı (K TL)")
ax4.set_title("Günlük İşlem Hacmi\n(Alım Event Window)")
ax4.set_xticks(range(-X_DAYS, X_DAYS+1))
ax4.set_xticklabels([str(d) if d%2==0 else "" for d in range(-X_DAYS, X_DAYS+1)], fontsize=8)
ax4.legend(handles=[mpatches.Patch(color=YON_RENK["Giriş"],label=f"Pre"),
                    mpatches.Patch(color="#1D4ED8",label="Post"),
                    Line2D([0],[0],color="#F59E0B",lw=2,ls="--",label="Alım")],
           loc="upper left", fontsize=8); style_axes(ax4)

for i, seg in enumerate(SEGMENT_SIRASI):
    net_pre  = NET_PRE_BUY.get(seg, 0) / 1e3
    net_post = NET_POST_BUY.get(seg, 0) / 1e3
    ax5.plot([0, 1], [net_pre, net_post], "o-", color=SEG_RENK[seg], lw=1.8, ms=5, alpha=0.88)
    ax5.text(-0.08, net_pre,  f"{SEG_SHORT[i]}", ha="right", va="center", fontsize=7, color=SEG_RENK[seg])
    ax5.text(1.08,  net_post, f"{net_post:.0f}K", ha="left", va="center", fontsize=7, color=SEG_RENK[seg])
ax5.axhline(0, ls="--", color=SPINE_CLR, lw=1.0, alpha=0.7)
ax5.set_xticks([0,1]); ax5.set_xticklabels(["Pre-Buy","Post-Buy"], fontsize=9)
ax5.set_ylabel("Ort. Net Akış (K TL)"); ax5.set_title("Net Akış Değişimi\n(Pre-Buy → Post-Buy)")
ax5.set_xlim(-0.45, 1.45); style_axes(ax5)

sns.heatmap(GECIS_ALIM, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, linecolor=GRID_CLR, ax=ax6,
            cbar_kws={"shrink":0.85, "label":"Geçiş %"},
            xticklabels=URUNLER, yticklabels=URUNLER)
ax6.set_title("Pre→Post-Buy Ürün Geçiş\n(Satır-Normalize %)"); ax6.set_xlabel("Sonraki Ürün"); ax6.set_ylabel("Önceki Ürün")
ax6.tick_params(labelsize=8.5)

style_fig(fig, "B — Fon Alım Davranışı  ·  Segment × Dönem Bazlı",
          f"Pre/Post Alım Analizi  |  Ürün Mix · Giriş Oranı · Net Akış · Geçiş Matrisi  ·  2 dönem")
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()


# ══════════════════════════════════════════════════════════════════
# § C — FON SATIM DAVRANIŞI  (4 panel)
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 10))
gs  = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.40)
ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0]); ax4 = fig.add_subplot(gs[1,1])

bottom = np.zeros(_N)
for urun in URUNLER:
    vals = URUN_PRE_SELL[urun].values
    ax1.bar(xs, vals, bottom=bottom, width=0.55, label=urun,
            color=URUN_RENKLER[urun], alpha=0.9, zorder=3, edgecolor="white", linewidth=0.5)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 9: ax1.text(xs[i], b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=6.5, fontweight="bold", color="white")
    bottom += vals
ax1.set_xticks(xs); ax1.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax1.set_title(f"Pre-Sell Ürün Mix\n(Satım öncesi −{X_DAYS} gün)")
ax1.set_ylabel("İşlem Dağılımı (%)"); ax1.set_ylim(0,115)
ax1.legend(loc="upper right", fontsize=8); style_axes(ax1)

bottom = np.zeros(_N)
for urun in URUNLER:
    vals = URUN_POST_SELL[urun].values
    ax2.bar(xs, vals, bottom=bottom, width=0.55, label=urun,
            color=URUN_RENKLER[urun], alpha=0.9, zorder=3, edgecolor="white", linewidth=0.5)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 9: ax2.text(xs[i], b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=6.5, fontweight="bold", color="white")
    bottom += vals
ax2.set_xticks(xs); ax2.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax2.set_title(f"Post-Sell Ürün Mix\n(Satım sonrası +{X_DAYS} gün)")
ax2.set_ylabel("İşlem Dağılımı (%)"); ax2.set_ylim(0,115)
ax2.legend(loc="upper right", fontsize=8); style_axes(ax2)

ax3.bar(xs-w/2, GIRIS_PRE_SELL.values,  width=w, color=YON_RENK["Çıkış"], alpha=0.9,  label="Pre-Sell",  zorder=3)
ax3.bar(xs+w/2, GIRIS_POST_SELL.values, width=w, color=YON_RENK["Çıkış"], alpha=0.50, label="Post-Sell", zorder=3)
ax3.axhline(50, ls="--", color=SPINE_CLR, lw=1.0, alpha=0.7)
ax3.set_xticks(xs); ax3.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax3.set_ylabel("Giriş Oranı (%)"); ax3.set_ylim(0, 85)
ax3.set_title("Para Giriş Oranı\n(Pre-Sell vs Post-Sell)")
ax3.legend(fontsize=8); style_axes(ax3)

sns.heatmap(GECIS_SATIM, annot=True, fmt=".0f", cmap="Reds",
            linewidths=0.5, linecolor=GRID_CLR, ax=ax4,
            cbar_kws={"shrink":0.85,"label":"Geçiş %"},
            xticklabels=URUNLER, yticklabels=URUNLER)
ax4.set_title("Pre→Post-Sell Ürün Geçiş\n(Satır-Normalize %)"); ax4.set_xlabel("Sonraki"); ax4.set_ylabel("Önceki")
ax4.tick_params(labelsize=8.5)

style_fig(fig, "C — Fon Satım Davranışı  ·  Segment × Dönem Bazlı",
          "Pre/Post Satım Analizi  |  Ürün Mix · Giriş Oranı · Geçiş Matrisi  ·  2 dönem")
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()


# ══════════════════════════════════════════════════════════════════
# § D — DÖNEMSELANALİZLER + HEAVY B/S  (2 figura)
# ══════════════════════════════════════════════════════════════════

# D1 — Dönemsel Karşılaştırma
fig_d = plt.figure(figsize=(20, 10))
gs_d  = GridSpec(2, 2, figure=fig_d, hspace=0.52, wspace=0.40)
ax_d1 = fig_d.add_subplot(gs_d[0,0]); ax_d2 = fig_d.add_subplot(gs_d[0,1])
ax_d3 = fig_d.add_subplot(gs_d[1,0]); ax_d4 = fig_d.add_subplot(gs_d[1,1])

sns.heatmap(DONEM_ALIM_PIVOT, ax=ax_d1, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, linecolor=GRID_CLR, cbar_kws={"shrink":0.8,"label":"M TL"},
            xticklabels=DONEM_SIRASI,
            yticklabels=[s.replace("_","\n") for s in SEGMENT_SIRASI])
ax_d1.set_title("Dönem × Segment — Alım Hacmi (M TL)")
ax_d1.set_xlabel("Dönem"); ax_d1.set_ylabel("Segment")
ax_d1.tick_params(axis="x", labelsize=8.5, rotation=20)
ax_d1.tick_params(axis="y", labelsize=7)
ax_d1.title.set_color(TITLE_CLR)

sns.heatmap(DONEM_SATIM_PIVOT, ax=ax_d2, annot=True, fmt=".2f", cmap="Reds",
            linewidths=0.5, linecolor=GRID_CLR, cbar_kws={"shrink":0.8,"label":"M TL"},
            xticklabels=DONEM_SIRASI,
            yticklabels=[s.replace("_","\n") for s in SEGMENT_SIRASI])
ax_d2.set_title("Dönem × Segment — Satım Hacmi (M TL)")
ax_d2.set_xlabel("Dönem"); ax_d2.set_ylabel("Segment")
ax_d2.tick_params(axis="x", labelsize=8.5, rotation=20)
ax_d2.tick_params(axis="y", labelsize=7)
ax_d2.title.set_color(TITLE_CLR)

sns.heatmap(DONEM_NET_PIVOT, ax=ax_d3, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, linecolor=GRID_CLR,
            cbar_kws={"shrink":0.8,"label":"M TL (net)"},
            xticklabels=DONEM_SIRASI,
            yticklabels=[s.replace("_","\n") for s in SEGMENT_SIRASI])
ax_d3.set_title("Net Fon Akışı — Dönem × Segment\n(Alım − Satım M TL  ·  Yeşil=net alım)")
ax_d3.set_xlabel("Dönem"); ax_d3.set_ylabel("Segment")
ax_d3.tick_params(axis="x", labelsize=8.5, rotation=20)
ax_d3.tick_params(axis="y", labelsize=7)
ax_d3.title.set_color(TITLE_CLR)

sns.heatmap(DONEM_ISLEM_PIVOT, ax=ax_d4, annot=True, fmt=".1f", cmap="YlOrBr",
            linewidths=0.5, linecolor=GRID_CLR, cbar_kws={"shrink":0.8,"label":"M TL"},
            xticklabels=DONEM_SIRASI,
            yticklabels=[s.replace("_","\n") for s in SEGMENT_SIRASI])
ax_d4.set_title("Dönem İçi İşlem Yoğunluğu (M TL)")
ax_d4.set_xlabel("Dönem"); ax_d4.set_ylabel("Segment")
ax_d4.tick_params(axis="x", labelsize=8.5, rotation=20)
ax_d4.tick_params(axis="y", labelsize=7)
ax_d4.title.set_color(TITLE_CLR)

style_fig(fig_d, "D — Dönemsel Analizler  ·  Segment × Dönem  ·  2025.09 vs 2026.03",
          "Alım · Satım · Net Akış · İşlem Yoğunluğu  ·  Heatmap bazlı karşılaştırma")
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()

# D2 — Heavy Buyer/Seller
fig_g = plt.figure(figsize=(20, 10))
gs_g  = GridSpec(2, 2, figure=fig_g, hspace=0.52, wspace=0.42)
ax_g1 = fig_g.add_subplot(gs_g[0,0]); ax_g2 = fig_g.add_subplot(gs_g[0,1])
ax_g3 = fig_g.add_subplot(gs_g[1,0]); ax_g4 = fig_g.add_subplot(gs_g[1,1])

sns.heatmap(HEAVY_BUYER_SEG, ax=ax_g1, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, linecolor=GRID_CLR, cbar_kws={"shrink":0.8,"label":"%"},
            xticklabels=DONEM_SIRASI, yticklabels=[s.replace("_","\n") for s in SEGMENT_SIRASI])
ax_g1.set_title("Heavy Buyer Oranı\n(Top %25 Alım · Dönem × Segment %)")
ax_g1.set_xlabel("Dönem"); ax_g1.tick_params(axis="x",labelsize=8.5,rotation=20)
ax_g1.tick_params(axis="y",labelsize=7); ax_g1.title.set_color(TITLE_CLR)

_g2_x = np.arange(_N)
_g2_n = (HEAVY_BUYER_TUTAR["Normal_K"].values if "Normal_K" in HEAVY_BUYER_TUTAR.columns else np.ones(_N))
_g2_h = (HEAVY_BUYER_TUTAR["Heavy_K"].values  if "Heavy_K"  in HEAVY_BUYER_TUTAR.columns else np.ones(_N))
ax_g2.bar(_g2_x-0.2, np.maximum(_g2_n,0.01), width=0.38, color=SEG_RENK_LIST, alpha=0.45, label="Normal", zorder=3, edgecolor="white")
ax_g2.bar(_g2_x+0.2, np.maximum(_g2_h,0.01), width=0.38, color=SEG_RENK_LIST, alpha=0.92, label="Heavy",  zorder=3, edgecolor="white")
ax_g2.set_xticks(_g2_x); ax_g2.set_xticklabels(SEG_SHORT, rotation=35, ha="right", fontsize=7.5)
ax_g2.set_ylabel("Ort. Alım Tutarı (K TL)"); ax_g2.set_title("Heavy vs Normal Alıcı\n(Ort. Tutar)")
ax_g2.set_yscale("log")
ax_g2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}M" if x>=1e3 else f"{x:.0f}K"))
ax_g2.legend(fontsize=9); style_axes(ax_g2, grid=False)

sns.heatmap(HEAVY_SELLER_SEG, ax=ax_g3, annot=True, fmt=".0f", cmap="Reds",
            linewidths=0.5, linecolor=GRID_CLR, cbar_kws={"shrink":0.8,"label":"%"},
            xticklabels=DONEM_SIRASI, yticklabels=[s.replace("_","\n") for s in SEGMENT_SIRASI])
ax_g3.set_title("Heavy Seller Oranı\n(Top %25 Satım · Dönem × Segment %)")
ax_g3.set_xlabel("Dönem"); ax_g3.tick_params(axis="x",labelsize=8.5,rotation=20)
ax_g3.tick_params(axis="y",labelsize=7); ax_g3.title.set_color(TITLE_CLR)

for j, (seg, short) in enumerate(zip(SEGMENT_SIRASI, SEG_SHORT)):
    _vals = [HEAVY_ISLEM_DONEM.loc[seg, d] if (seg in HEAVY_ISLEM_DONEM.index and d in HEAVY_ISLEM_DONEM.columns) else 0 for d in DONEM_SIRASI]
    ax_g4.plot(range(_ND), _vals, "o-", color=SEG_RENK[seg], lw=1.8, ms=5, alpha=0.88, label=short)
ax_g4.set_xticks(range(_ND)); ax_g4.set_xticklabels(DONEM_SIRASI, fontsize=9, rotation=15)
ax_g4.set_ylabel("Ort. İşlem Tutarı (K TL)"); ax_g4.set_title("Heavy Buyer İşlem Trendi\n(Dönem bazlı)")
ax_g4.legend(title="Segment", loc="upper left", fontsize=7, ncol=5); style_axes(ax_g4)

style_fig(fig_g, "D2 — Heavy Buyer / Seller Segment Analizi  ·  2 Dönem",
          f"Top %25 alım/satım müşteri profili  |  {len(_heavy_ids):,} heavy buyer · {_N} segment")
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()


# ══════════════════════════════════════════════════════════════════
# § E — SEKANSİYEL İŞLEM AĞLARI
# ══════════════════════════════════════════════════════════════════
def draw_seq_net(ax, G, title, min_pct=NET_SEQ_MIN_PCT, show_pct=True):
    ax.set_facecolor(FIG_BG); ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", color=TITLE_CLR, pad=8)
    if G is None or G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "Yeterli veri yok", ha="center", va="center",
                transform=ax.transAxes, color=SPINE_CLR, fontsize=9); return
    try: pos = nx.spring_layout(G, weight="pct", seed=42, k=3.2, iterations=200)
    except: pos = nx.circular_layout(G)
    vis = [(u,v,d) for u,v,d in G.edges(data=True) if d.get("pct",0)>=min_pct and u!=v]
    self_lp = [(u,u,G[u][u]) for u in G.nodes if G.has_edge(u,u) and G[u][u].get("pct",0)>=min_pct]
    if not vis: vis = [(u,v,d) for u,v,d in G.edges(data=True) if u!=v]
    max_pct  = max((d.get("pct",1) for _,_,d in vis), default=1)
    max_freq = max((G.nodes[n].get("freq",1) for n in G.nodes), default=1)
    for node, _, edge_d in self_lp:
        xy = pos[node]; clr = G.nodes[node].get("color","#888")
        off = 0.22
        ax.annotate("", xy=(xy[0]+off,xy[1]+off*0.5), xytext=(xy[0]+off*0.5,xy[1]+off),
                    arrowprops=dict(arrowstyle="-|>",connectionstyle="arc3,rad=0.80",
                                    color=clr,lw=1.8,alpha=0.68), zorder=4)
    for u,v,d in vis:
        pct = d.get("pct",1); lw = 0.8+(pct/max_pct)*4.5; alpha=0.25+(pct/max_pct)*0.65
        clr = G.nodes[u].get("color","#888")
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>",color=clr,lw=lw,alpha=alpha,
                                    connectionstyle="arc3,rad=0.18"), zorder=3)
        if show_pct:
            mx=(pos[u][0]+pos[v][0])/2; my=(pos[u][1]+pos[v][1])/2
            dx=pos[v][0]-pos[u][0]; dy=pos[v][1]-pos[u][1]; dist=max(_math.hypot(dx,dy),0.001)
            ox,oy=-dy/dist*0.09,dx/dist*0.09
            ax.text(mx+ox,my+oy,f"{pct:.0f}%",fontsize=6.5,ha="center",va="center",
                    color=clr,fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.10",fc="white",ec="none",alpha=0.82),zorder=5)
    for node in G.nodes:
        xy = pos[node]; freq = max(G.nodes[node].get("freq",1),1)
        r = 0.065+(freq/max_freq)*0.10; color = G.nodes[node].get("color","#888")
        ax.add_patch(plt.Circle(xy,r+0.022,color=color,alpha=0.16,zorder=6))
        ax.add_patch(plt.Circle(xy,r,color=color,alpha=0.92,zorder=7))
        ax.text(xy[0],xy[1],node,ha="center",va="center",fontsize=8,fontweight="bold",color="white",zorder=8)
    xs_p=[p[0] for p in pos.values()]; ys_p=[p[1] for p in pos.values()]; pad=0.38
    ax.set_xlim(min(xs_p)-pad,max(xs_p)+pad); ax.set_ylim(min(ys_p)-pad,max(ys_p)+pad)
    ax.set_aspect("equal")

fig1 = plt.figure(figsize=(20, 12))
gs1  = GridSpec(2, 4, figure=fig1, hspace=0.44, wspace=0.34)
fig1.patch.set_facecolor(FIG_BG)
ax_gen=fig1.add_subplot(gs1[0,0]); ax_pb=fig1.add_subplot(gs1[0,1])
ax_postb=fig1.add_subplot(gs1[0,2]); ax_hm=fig1.add_subplot(gs1[0,3])
ax_prs=fig1.add_subplot(gs1[1,0]); ax_posts=fig1.add_subplot(gs1[1,1])
ax_dhm=fig1.add_subplot(gs1[1,2]); ax_bg=fig1.add_subplot(gs1[1,3])

draw_seq_net(ax_gen,   G_SEQ_ALL,       f"Genel Sekansiyel Ağ\n({len(_pairs_all):,} geçiş)")
draw_seq_net(ax_pb,    G_SEQ_PRE_BUY,   f"Fon Alımı Öncesi\n(−{X_DAYS} gün)")
draw_seq_net(ax_postb, G_SEQ_POST_BUY,  f"Fon Alımı Sonrası\n(+{X_DAYS} gün)")
draw_seq_net(ax_prs,   G_SEQ_PRE_SELL,  f"Fon Satımı Öncesi\n(−{X_DAYS} gün)")
draw_seq_net(ax_posts, G_SEQ_POST_SELL, f"Fon Satımı Sonrası\n(+{X_DAYS} gün)")

sns.heatmap(TRANS_MAT_ALL*100, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, linecolor=GRID_CLR, ax=ax_hm,
            annot_kws={"size":9,"weight":"bold"}, cbar_kws={"shrink":0.82,"label":"Geçiş %"},
            xticklabels=URUNLER, yticklabels=URUNLER)
ax_hm.set_title("Geçiş Olasılık Matrisi\n(Tüm Müşteriler · Satır %)", fontsize=10, color=TITLE_CLR)

_delta = (TRANS_MAT_POST_BUY - TRANS_MAT_PRE_BUY)*100
_vmax  = max(abs(_delta.values).max(), 0.5)
sns.heatmap(_delta, annot=True, fmt="+.0f", cmap="RdYlGn", vmin=-_vmax, vmax=_vmax, center=0,
            linewidths=0.5, linecolor=GRID_CLR, ax=ax_dhm,
            annot_kws={"size":9}, cbar_kws={"shrink":0.82,"label":"Δ%"},
            xticklabels=URUNLER, yticklabels=URUNLER)
ax_dhm.set_title("Post-Buy − Pre-Buy Δ\n(Yeşil=artan geçiş)", fontsize=10, color=TITLE_CLR)

if not TOP_BIGRAMS.empty:
    _bg = TOP_BIGRAMS.head(12)
    _bg_clrs = [URUN_RENKLER.get(r.split(" → ")[0],"#888") for r in _bg["label"]]
    ax_bg.barh(range(len(_bg)),_bg["count"].values,color=_bg_clrs,alpha=0.85,edgecolor="white")
    ax_bg.set_yticks(range(len(_bg))); ax_bg.set_yticklabels(_bg["label"].values,fontsize=8.5)
    ax_bg.invert_yaxis()
ax_bg.set_title("En Sık 12 İkili Geçiş",fontsize=10,color=TITLE_CLR)
ax_bg.set_xlabel("Frekans",fontsize=8.5); style_axes(ax_bg)

style_fig(fig1, "E — Sekansiyel İşlem Ağı · Geçiş Olasılıkları  ·  2 Dönem",
          f"{len(_pairs_all):,} geçiş · eşik ≥{NET_SEQ_MIN_EDGE} · görsel ≥%{NET_SEQ_MIN_PCT:.0f}")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# Segment ağları
_ncols_seg = min(len(SEGMENT_SIRASI), 4)
_nrows_seg = _math.ceil(len(SEGMENT_SIRASI)/_ncols_seg)
fig2 = plt.figure(figsize=(18, 5.5*_nrows_seg + 1.5)); fig2.patch.set_facecolor(FIG_BG)
gs2 = GridSpec(_nrows_seg, _ncols_seg, figure=fig2, hspace=0.52, wspace=0.36)
for _idx, _seg in enumerate(SEGMENT_SIRASI):
    _ax = fig2.add_subplot(gs2[_idx//_ncols_seg, _idx%_ncols_seg])
    draw_seq_net(_ax, G_SEQ_SEG.get(_seg), f"{SEG_SHORT[_idx]}", min_pct=1.5)
    for _sp in _ax.spines.values():
        _sp.set_edgecolor(SEG_RENK[_seg]); _sp.set_linewidth(2.4); _sp.set_visible(True)
    _ax.set_facecolor(AXES_BG)
style_fig(fig2, "E2 — Segment Bazlı Sekansiyel Ağlar  ·  2 Dönem")
plt.tight_layout(rect=[0,0,1,0.965])
plt.show()


# ══════════════════════════════════════════════════════════════════
# § F — STRATEJİK İÇGÖRÜLER
# ══════════════════════════════════════════════════════════════════
en_aktif_seg   = DAVRANIS_SKOR["Aktivite_Skoru"].idxmax()
en_sadik_seg   = SADAKAT["Cok_Donem_Pct"].idxmax()
en_yuksek_aum  = AUM.idxmax()
alim_artis_seg = (GIRIS_POST_BUY - GIRIS_PRE_BUY).idxmax()
pre_buy_vadesiz_ratio = URUN_PRE_BUY["Vadesiz"].mean() if "Vadesiz" in URUN_PRE_BUY.columns else 0
post_sell_doviz_ratio = URUN_POST_SELL["Döviz"].mean()  if "Döviz"   in URUN_POST_SELL.columns else 0
gecis_diagonal_alim   = np.diag(GECIS_ALIM.values).mean()

print("="*72)
print("  BANKA FON MÜŞTERİ DAVRANIŞ ANALİZİ  —  2025.09 & 2026.03")
print("="*72)
print(f"""
  TEMEL KPI'LAR
  ─────────────────────────────────────────────────────
  Toplam Fon Alım   : {KPI['toplam_alim_m']:>7.1f} M TL
  Toplam Fon Satım  : {KPI['toplam_satim_m']:>7.1f} M TL
  Net AUM           : {KPI['net_aum_m']:>7.1f} M TL
  Toplam Olay       : {KPI['toplam_event']:>7,}
  Penetrasyon Ort.  : %{KPI['penetrasyon_ort']:.1f}
  Sadakat Ort.      : %{KPI['sadakat_ort']:.1f}
  ─────────────────────────────────────────────────────
  En Aktif Segment  : {en_aktif_seg}  (Skor: {DAVRANIS_SKOR.loc[en_aktif_seg,'Aktivite_Skoru']:.0f}/100)
  En Sadık Segment  : {en_sadik_seg}  (%{SADAKAT.loc[en_sadik_seg,'Cok_Donem_Pct']:.0f} her iki döneme katıldı)
  En Yüksek AUM     : {en_yuksek_aum}  ({AUM.get(en_yuksek_aum,0)/1e6:.1f}M TL)
  Pre-Buy Vadesiz % : {pre_buy_vadesiz_ratio:.1f}%  (nakit birikim sinyali)
  Post-Sell Döviz % : {post_sell_doviz_ratio:.1f}%  (re-investment kur riski)
  Geçiş kalıcılığı  : {gecis_diagonal_alim:.0f}%  (aynı ürün grubu devamı)
""")
for seg in SEGMENT_SIRASI:
    pen = PENETRASYON.loc[seg,"Alim_Pct"]; sad = SADAKAT.loc[seg,"Cok_Donem_Pct"]
    delta = GIRIS_POST_BUY.get(seg,0) - GIRIS_PRE_BUY.get(seg,0)
    print(f"  [{seg}]  Pen %{pen:.0f} · Sadakat %{sad:.0f} · Δ{delta:+.1f}")
print(f"\n  ✅ Analiz: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")


# ══════════════════════════════════════════════════════════════════
# § G — t-1 BAKİYE HEATMAPLER
# ══════════════════════════════════════════════════════════════════
if not ALIM_BAK_HM.empty:
    _yt = [SEG_SHORT[SEGMENT_SIRASI.index(s)] for s in SEGMENT_SIRASI]
    _hm_base = dict(linewidths=0.4, linecolor="#E2E8F0", yticklabels=_yt)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    style_fig(fig, "G — t-1 Dönemi Bakiyesine Göre Alım / Satım Oranları  ·  2025.09→2026.03",
              "Alım/Satım tutarının t-1 bakiyesine oranı (Medyan %)  ·  İlk dönem = referans")

    ax = axes[0,0]
    _vmax_a = max(float(ALIM_BAK_HM.values.max()), 0.1)
    sns.heatmap(ALIM_BAK_HM, ax=ax, cmap="Blues", vmin=0, vmax=_vmax_a,
                annot=True, fmt=".1f", cbar_kws={"label":"Alım/Bakiye(%)","shrink":0.78}, **_hm_base)
    ax.set_title("Alım / t-1 Bakiye (%) · Medyan"); ax.set_xlabel("Dönem")
    ax.tick_params(axis="x", rotation=15, labelsize=8.5); ax.tick_params(axis="y", labelsize=7.5)

    ax = axes[0,1]
    _vmax_s = max(float(SATIM_BAK_HM.values.max()), 0.1)
    sns.heatmap(SATIM_BAK_HM, ax=ax, cmap="Reds", vmin=0, vmax=_vmax_s,
                annot=True, fmt=".1f", cbar_kws={"label":"Satım/Bakiye(%)","shrink":0.78}, **_hm_base)
    ax.set_title("Satım / t-1 Bakiye (%) · Medyan"); ax.set_xlabel("Dönem")
    ax.tick_params(axis="x", rotation=15, labelsize=8.5); ax.tick_params(axis="y", labelsize=7.5)

    ax = axes[1,0]
    _amax = max(float(abs(ALIM_BAK_DELTA.values).max()), 0.1)
    sns.heatmap(ALIM_BAK_DELTA, ax=ax, cmap="RdYlGn", center=0, vmin=-_amax, vmax=_amax,
                annot=True, fmt=".1f", cbar_kws={"label":"Δ pp","shrink":0.78}, **_hm_base)
    ax.set_title("Alım/Bakiye Değişimi (pp · ΔPoP)"); ax.set_xlabel("Dönem")
    ax.tick_params(axis="x", rotation=15, labelsize=8.5); ax.tick_params(axis="y", labelsize=7.5)

    ax = axes[1,1]
    _smax = max(float(abs(SATIM_BAK_DELTA.values).max()), 0.1)
    sns.heatmap(SATIM_BAK_DELTA, ax=ax, cmap="RdYlGn", center=0, vmin=-_smax, vmax=_smax,
                annot=True, fmt=".1f", cbar_kws={"label":"Δ pp","shrink":0.78}, **_hm_base)
    ax.set_title("Satım/Bakiye Değişimi (pp · ΔPoP)"); ax.set_xlabel("Dönem")
    ax.tick_params(axis="x", rotation=15, labelsize=8.5); ax.tick_params(axis="y", labelsize=7.5)

    plt.show()
    print("✅ t-1 Bakiye heatmapler gösterildi")
else:
    print("⚠️  t-1 hesaplama için yeterli dönem yok (ilk dönemde t-1 bakiye yoktur).")


# ══════════════════════════════════════════════════════════════════
# § H — FON-GEÇMEYEN SEKANS HEATMAPLER  (2 dönem)
# ══════════════════════════════════════════════════════════════════
_yt  = [SEG_SHORT[SEGMENT_SIRASI.index(s)] for s in SEGMENT_SIRASI]
_xt  = [FG_TURLER_KISALT.get(t,t) for t in FG_TURLER]
_hm_kw = dict(linewidths=0.3, linecolor="#E2E8F0", annot=True, fmt=".0f", yticklabels=_yt)

fig1, axes1 = plt.subplots(2, 2, figsize=(22, 14), constrained_layout=True)
style_fig(fig1, "H — Fon-Geçmeyen İşlem Türleri — Pre/Post Alım & Satım  (Segment · %)",
          "Mavi=Pre-Buy · Yeşil=Post-Buy · Turuncu=Pre-Sell · Mor=Post-Sell  ·  2 dönem konsolide")

_panels = [
    (axes1[0,0], DIST_PRE_BUY,   "#1D4ED8", "Pre-Buy  — Fon-Geçmeyen İşlem Türü (%)"),
    (axes1[0,1], DIST_POST_BUY,  "#059669", "Post-Buy — Fon-Geçmeyen İşlem Türü (%)"),
    (axes1[1,0], DIST_PRE_SELL,  "#D97706", "Pre-Sell  — Fon-Geçmeyen İşlem Türü (%)"),
    (axes1[1,1], DIST_POST_SELL, "#7C3AED", "Post-Sell — Fon-Geçmeyen İşlem Türü (%)"),
]
for ax, data, base_color, ttl in _panels:
    cmap_custom = sns.light_palette(base_color, as_cmap=True)
    sns.heatmap(data, ax=ax, cmap=cmap_custom, cbar_kws={"label":"% payı","shrink":0.78}, **_hm_kw)
    ax.set_title(ttl); ax.set_xticklabels(_xt, rotation=30, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
plt.show()

# Dönem bazlı 2 sütun
_empty = pd.DataFrame(0.0, index=SEGMENT_SIRASI, columns=FG_TURLER)
fig2, axes2 = plt.subplots(2, 2, figsize=(22, 12), constrained_layout=True)
style_fig(fig2, "H2 — Fon-Geçmeyen Sekans — Dönem Bazlı Post−Pre Farkı (pp)  ·  2025.09 & 2026.03",
          "Yeşil=Post sonrası artış · Kırmızı=azalış  |  Üst: Alım · Alt: Satım")

for col_i, d in enumerate(DONEM_SIRASI):
    diff_buy = (DIST_POST_BUY_D.get(d,_empty) - DIST_PRE_BUY_D.get(d,_empty)).fillna(0)
    _amax_b  = max(float(abs(diff_buy.values).max()), 0.1)
    ax = axes2[0, col_i]
    sns.heatmap(diff_buy, ax=ax, cmap="RdYlGn", center=0, vmin=-_amax_b, vmax=_amax_b,
                annot=True, fmt=".0f", linewidths=0.3, linecolor="#E2E8F0",
                yticklabels=(_yt if col_i==0 else False))
    ax.set_title(f"{d}  ·  Post-Buy − Pre-Buy (pp)", fontsize=9)
    ax.set_xticklabels(_xt, rotation=35, ha="right", fontsize=7)
    if col_i==0: ax.set_ylabel("Alım Sekansı", fontsize=9, fontweight="bold")

    diff_sell = (DIST_POST_SELL_D.get(d,_empty) - DIST_PRE_SELL_D.get(d,_empty)).fillna(0)
    _amax_s   = max(float(abs(diff_sell.values).max()), 0.1)
    ax = axes2[1, col_i]
    sns.heatmap(diff_sell, ax=ax, cmap="PuOr", center=0, vmin=-_amax_s, vmax=_amax_s,
                annot=True, fmt=".0f", linewidths=0.3, linecolor="#E2E8F0",
                yticklabels=(_yt if col_i==0 else False))
    ax.set_title(f"{d}  ·  Post-Sell − Pre-Sell (pp)", fontsize=9)
    ax.set_xticklabels(_xt, rotation=35, ha="right", fontsize=7)
    if col_i==0: ax.set_ylabel("Satım Sekansı", fontsize=9, fontweight="bold")

plt.show()
print("✅ Fon-geçmeyen sekans analizi tamamlandı")


# ══════════════════════════════════════════════════════════════════
# § I — PROFESYONEL KOMPOZİT NETWORK  (Yapılandırılmış Hiyerarşik)
# ══════════════════════════════════════════════════════════════════

def _structured_layout(G):
    """
    Yapılandırılmış ızgara düzeni:
    - X ekseni: Ürün grubu (Vadesiz=0, Vadeli=1, Yatırım=2, Döviz=3, Kredi=4) * X_SPACING
    - Y ekseni: +3.5 = Giriş, -3.5 = Çıkış
    - Aynı ürün+yön grubundaki node'lar yatayda yayılır
    """
    from collections import defaultdict
    URUN_ORD  = ["Vadesiz","Vadeli","Yatırım","Döviz","Kredi"]
    X_SPACING = 5.5
    PROD_X    = {p: i * X_SPACING for i, p in enumerate(URUN_ORD)}

    groups = defaultdict(list)
    for node in G.nodes():
        parts = str(node).split("|")
        prod  = parts[0] if parts else "Other"
        yon   = parts[2] if len(parts) > 2 else "Other"
        groups[(prod, yon)].append(node)

    pos = {}
    for (prod, yon), nodes in groups.items():
        x_base = PROD_X.get(prod, len(URUN_ORD) * X_SPACING)
        y_base = 3.5 if yon == "Giriş" else -3.5
        y_dir  = 1   if yon == "Giriş" else -1
        n      = len(nodes)
        for j, node in enumerate(sorted(nodes)):
            x_off = (j - (n - 1) / 2.0) * 1.3
            y_off = j * 1.8 * y_dir if n > 1 else 0
            pos[node] = (x_base + x_off, y_base + y_off)
    for node in G.nodes():
        if node not in pos:
            pos[node] = (len(URUN_ORD) * X_SPACING + 2, 0.0)
    return pos


def _bezier_pts(x0, y0, x1, y1, n=50, strength=0.38):
    """Cubic Bezier eğri noktaları üretir."""
    dx, dy = x1 - x0, y1 - y0
    dist   = max((dx**2 + dy**2)**0.5, 1e-9)
    nx_, ny_ = -dy / dist, dx / dist
    cx0 = x0 + dx*0.33 + nx_ * dist * strength
    cy0 = y0 + dy*0.33 + ny_ * dist * strength
    cx1 = x0 + dx*0.67 + nx_ * dist * strength
    cy1 = y0 + dy*0.67 + ny_ * dist * strength
    t   = np.linspace(0, 1, n)
    bx  = (1-t)**3*x0 + 3*t*(1-t)**2*cx0 + 3*t**2*(1-t)*cx1 + t**3*x1
    by  = (1-t)**3*y0 + 3*t*(1-t)**2*cy0 + 3*t**2*(1-t)*cy1 + t**3*y1
    return bx.tolist(), by.tolist()


def build_professional_network(G, title="Profesyonel İşlem Ağı",
                                max_nodes=32, top_arrows=50):
    """
    Yapılandırılmış hiyerarşik layout:
    • Ürünler X ekseninde ayrışır (çakışma yok)
    • Giriş/Çıkış Y ekseninde ayrışır
    • Kenarlar Bezier eğrileri (iç içe değil)
    • Beyaz daire + renkli kenarlık
    • Sade ızgara çizgileri, ürün başlıkları, yön etiketleri
    """
    URUN_ORD  = ["Vadesiz","Vadeli","Yatırım","Döviz","Kredi"]
    X_SPACING = 5.5
    PROD_X    = {p: i * X_SPACING for i, p in enumerate(URUN_ORD)}

    if G.number_of_nodes() == 0:
        return go.Figure().update_layout(title=title, paper_bgcolor="white")

    # Subgraph with top-N nodes
    top_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n].get("freq",0), reverse=True)[:max_nodes]
    Gs = G.subgraph(top_nodes).copy()

    # PageRank
    try:
        pr = nx.pagerank(Gs, weight="weight", max_iter=300) if Gs.number_of_edges()>0 else {n:1/max(Gs.number_of_nodes(),1) for n in Gs.nodes()}
    except:
        pr = {n: Gs.degree(n) for n in Gs.nodes()}
    nx.set_node_attributes(Gs, pr, "_pr")

    pos = _structured_layout(Gs)
    traces = []

    # ── Kenar izleri (Bezier eğrisi) ──────────────────────────────
    all_edges   = sorted(Gs.edges(data=True), key=lambda e: e[2].get("weight",0), reverse=True)
    max_w       = max((e[2].get("weight",0) for e in all_edges), default=1)

    for src, dst, edata in all_edges:
        if src not in pos or dst not in pos: continue
        x0, y0 = pos[src]; x1, y1 = pos[dst]
        w     = edata.get("weight", 1)
        pct   = edata.get("pct", 0.0)
        alpha = round(max(0.15, min(0.72, 0.15 + (w/max_w)*0.57)), 2)
        width = round(max(0.8, (w/max_w)*6.5), 2)
        src_prod = str(src).split("|")[0]
        clr  = URUN_RENKLER.get(src_prod, "#94A3B8")

        if src == dst:  # self-loop: küçük çember
            theta = np.linspace(0, 2*np.pi, 44)
            r = 0.75
            lx = (x0 + r * np.cos(theta)).tolist()
            ly = (y0 + r * np.sin(theta)).tolist()
        else:
            lx, ly = _bezier_pts(x0, y0, x1, y1, n=50, strength=0.35)

        traces.append(go.Scatter(
            x=lx, y=ly, mode="lines",
            line=dict(width=width, color=f"rgba{tuple(int(clr.lstrip('#')[i:i+2],16) for i in (0,2,4))+( alpha,)}"),
            hovertemplate=(f"<b>{str(src).replace('|',' | ')}</b> → "
                           f"<b>{str(dst).replace('|',' | ')}</b><br>"
                           f"Geçiş: {int(w):,}  Pay: {pct:.1f}%<extra></extra>"),
            showlegend=False,
        ))

    # ── Ok başlıkları ─────────────────────────────────────────────
    arrows = []
    for src, dst, edata in all_edges[:top_arrows]:
        if src == dst or src not in pos or dst not in pos: continue
        x0,y0 = pos[src]; x1,y1 = pos[dst]
        lx, ly = _bezier_pts(x0, y0, x1, y1, n=50, strength=0.35)
        if len(lx) < 4: continue
        axe, aye = lx[-3], ly[-3]
        bxe, bye = lx[-1], ly[-1]
        adx, ady = bxe-axe, bye-aye
        adist = max((adx**2+ady**2)**0.5, 1e-9)
        node_r = 0.55
        bxe_s = bxe - adx/adist*node_r
        bye_s = bye - ady/adist*node_r
        src_prod = str(src).split("|")[0]
        clr = URUN_RENKLER.get(src_prod, "#94A3B8")
        w   = edata.get("weight",1)
        arrows.append(dict(
            x=bxe_s, y=bye_s, ax=axe, ay=aye,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=4, arrowsize=0.9,
            arrowwidth=max(0.5, (w/max_w)*3.0),
            arrowcolor=clr + "CC",
        ))

    # ── Node'lar (ürün grubuna göre grupla) ───────────────────────
    from collections import defaultdict as _dd
    urun_groups = _dd(list)
    for n in Gs.nodes():
        urun_groups[str(n).split("|")[0]].append(n)

    node_freqs = {n: Gs.nodes[n].get("freq",1) for n in Gs.nodes()}
    max_freq   = max(node_freqs.values(), default=1)

    for urun in URUN_ORD + ["Other"]:
        nlist = urun_groups.get(urun, [])
        if not nlist: continue
        border_clr = URUN_RENKLER.get(urun, "#94A3B8")
        sizes = [round(42 + 46*(node_freqs[n]/max_freq)**0.6) for n in nlist]
        hover = []
        for n in nlist:
            parts = n.split("|")
            hover.append(f"<b>{n.replace('|',' | ')}</b><br>"
                         f"Ürün: {parts[0]}<br>"
                         f"Tür: {parts[1] if len(parts)>1 else ''}<br>"
                         f"Yön: {parts[2] if len(parts)>2 else ''}<br>"
                         f"Frekans: {node_freqs[n]:,}<br>"
                         f"PageRank: {pr.get(n,0):.4f}")
        labels = [_short_comp_label(n) for n in nlist]
        traces.append(go.Scatter(
            x=[pos[n][0] for n in nlist],
            y=[pos[n][1] for n in nlist],
            mode="markers+text",
            name=urun,
            text=labels,
            textposition="middle center",
            textfont=dict(size=8, color=border_clr, family="DejaVu Sans"),
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(
                size=sizes, sizemode="diameter",
                color="white",
                line=dict(color=border_clr, width=3.2),
                symbol="circle",
            ),
        ))

    # ── Arka plan şekilleri ────────────────────────────────────────
    shapes = []
    for i, prod in enumerate(URUN_ORD):
        x_c = PROD_X[prod]
        # Ürün grup bandı (çok hafif arkaplan)
        shapes.append(dict(
            type="rect",
            x0=x_c - 2.0, x1=x_c + 2.0, y0=-8.5, y1=8.5,
            fillcolor=URUN_RENKLER[prod] + "0A",
            line=dict(width=0),
        ))
        if i > 0:  # Dikey ayırıcı çizgiler
            shapes.append(dict(
                type="line",
                x0=x_c - 2.5, x1=x_c - 2.5, y0=-8.5, y1=8.5,
                line=dict(color="#E2E8F0", width=1.2, dash="dot"),
            ))
    # Giriş/Çıkış yatay ayırıcı
    shapes.append(dict(
        type="line",
        x0=-3, x1=PROD_X.get("Kredi", 20) + 3, y0=0, y1=0,
        line=dict(color="#CBD5E1", width=1.5, dash="dash"),
    ))

    # ── Annotasyonlar ─────────────────────────────────────────────
    annotations = list(arrows)
    annotations += [
        dict(x=-2.5, y=7.8, text="<b>↑ GİRİŞ İŞLEMLERİ</b>", showarrow=False,
             font=dict(size=12, color="#059669"), xanchor="left"),
        dict(x=-2.5, y=-7.8, text="<b>↓ ÇIKIŞ İŞLEMLERİ</b>", showarrow=False,
             font=dict(size=12, color="#DC2626"), xanchor="left"),
    ]
    for prod in URUN_ORD:
        annotations.append(dict(
            x=PROD_X[prod], y=9.5, text=f"<b>{prod}</b>",
            showarrow=False,
            font=dict(size=13, color=URUN_RENKLER[prod]),
            xanchor="center",
        ))

    xs_all  = [p[0] for p in pos.values()]
    ys_all  = [p[1] for p in pos.values()]
    x_pad = 3; y_pad = 1.5

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=15, family="DejaVu Sans", color="#0F172A")),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=True,
        legend=dict(title=dict(text="Ürün Grubu", font=dict(size=11)),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#E2E8F0", borderwidth=1,
                    x=1.01, y=0.98, font=dict(size=10)),
        xaxis=dict(visible=False, range=[min(xs_all)-x_pad, max(xs_all)+x_pad]),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1,
                   range=[min(ys_all)-y_pad, max(ys_all)+y_pad]),
        height=780, width=1200,
        shapes=shapes, annotations=annotations,
        margin=dict(l=30, r=220, t=90, b=40),
    )
    return fig


# ── Tüm segment kompozit network ──────────────────────────────────
print("\n▶  Profesyonel kompozit network oluşturuluyor...")
fig_net = build_professional_network(
    G_COMPOSITE,
    title=(
        "Tüm Segmentler — Kompozit İşlem Akış Networkü  ·  2025.09 & 2026.03<br>"
        "<sup>Her node = Ürün | İşlem Türü | Yön  ·  Boyut ∝ Frekans  ·  "
        "Kenar kalınlığı ∝ Geçiş sıklığı  ·  Bezier eğrili kenarlar</sup>"
    ),
)
fig_net.show()

HTML_DIR = "."
HTML_ALL = os.path.join(HTML_DIR, "islem_network_tum.html")
fig_net.write_html(HTML_ALL, include_plotlyjs="cdn", full_html=True,
                   config={"scrollZoom": True, "displayModeBar": True})
_abs_all = os.path.abspath(HTML_ALL)
print(f"✅ HTML kaydedildi: {_abs_all}")

display(HTML(f"""
<div style="padding:12px 16px;background:#F0F9FF;border:1px solid #BAE6FD;
            border-radius:10px;display:inline-block;margin:8px 0">
  <a href="{HTML_ALL}" target="_blank"
     style="font-size:15px;font-weight:bold;color:#0369A1;text-decoration:none">
    🌐 Profesyonel Network Grafiğini Aç (Tüm Segmentler)
  </a>
  <span style="margin-left:12px;color:#64748B;font-size:11px">{_abs_all}</span>
</div>
"""))

# ── Segment bazlı HTML'ler ──────────────────────────────────────────
print("\n▶  Segment bazlı profesyonel networkler...")
_seg_links = []
for seg in SEGMENT_SIRASI:
    Gs = G_COMPOSITE_SEG.get(seg)
    if Gs is None or Gs.number_of_nodes() == 0: continue
    figs = build_professional_network(
        Gs,
        title=f"{seg.replace('_',' ')} — İşlem Ağı  ·  2025.09 & 2026.03<br>"
              f"<sup>Yapılandırılmış hiyerarşik layout  ·  Bezier kenarlar</sup>",
    )
    fname = os.path.join(HTML_DIR, f"network_{seg}.html")
    figs.write_html(fname, include_plotlyjs="cdn", full_html=True,
                    config={"scrollZoom": True, "displayModeBar": True})
    _seg_links.append((seg, fname))
    print(f"  ✓ {seg}: {Gs.number_of_nodes()} node · {Gs.number_of_edges()} kenar")

_link_html = (
    '<div style="padding:14px;background:#F8FAFC;border:1px solid #E2E8F0;'
    'border-radius:10px;max-width:950px">'
    '<b style="color:#0F172A;font-size:13px">📂 Segment Bazlı Profesyonel Network HTML:</b>'
    '<br><br>')
for seg, fname in _seg_links:
    clr = SEG_RENK.get(seg, "#64748B")
    _link_html += (f'<a href="{fname}" target="_blank" '
                   f'style="display:inline-block;margin:4px 5px;padding:6px 14px;'
                   f'background:white;border:2px solid {clr};border-radius:7px;'
                   f'color:{clr};font-weight:bold;font-size:11px;text-decoration:none">'
                   f'{seg.replace("_"," ")}</a>')
_link_html += "</div>"
display(HTML(_link_html))
print(f"\n✅ {len(_seg_links)} segment network HTML hazır")
print(f"\n{'═'*60}")
print("  ✅  PART 2 TAMAMLANDI")
print(f"{'═'*60}")
print("  Chartlar    : A (Genel) · B (Alım) · C (Satım) · D (Dönemsel)")
print("  Ağlar       : E (Sekans) · I (Profesyonel Kompozit)")
print("  Heatmapler  : G (t-1 Bakiye) · H (Fon-Geçmeyen Sekans)")
print(f"  HTML        : {len(_seg_links)+1} dosya oluşturuldu")
