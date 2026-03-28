"""
visualize_phaseflow.py
======================
Visualizzazione della direzione media del flusso di fase EEG.

Uso:
    python visualize_phaseflow.py

Prerequisiti: aver già eseguito main_phaseflow.py nella stessa sessione,
oppure salvare results con np.save e ricaricarli qui.
Questo script va importato e chiamato dopo run_pipeline().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate elettrodi (stesse di main_phaseflow.py)
# ─────────────────────────────────────────────────────────────────────────────
HEAD_RADIUS_MM = 85.0
_COORDS = {
    "Fp1":(-18,72),"Fp2":(18,72),"F7":(-54,72),"F3":(-36,54),"FZ":(0,54),
    "T3":(-90,90),"C3":(-54,90),"CZ":(0,90),"C4":(54,90),"T4":(90,90),
    "T5":(-54,126),"P3":(-36,126),"PZ":(0,126),"P4":(36,126),"T6":(54,126),
    "O1":(-18,144),"O2":(18,144),"F4":(36,54),"F8":(54,72),
}
CHANNELS = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']

def sph_to_cart(th_deg, ph_deg, r=HEAD_RADIUS_MM):
    th, ph = np.radians(th_deg), np.radians(ph_deg)
    return np.array([r*np.sin(ph)*np.cos(th),
                     r*np.sin(ph)*np.sin(th),
                     r*np.cos(ph)])

loc = np.array([sph_to_cart(*_COORDS[ch]) for ch in CHANNELS])  # (19, 3)


# ─────────────────────────────────────────────────────────────────────────────
def plot_phaseflow(v: dict, sfreq: float = 200.0,
                  percentile_clip: float = 95.0,
                  save_path: str = None):
    """
    Visualizza la direzione media del flusso di fase.

    Parametri
    ----------
    v          : dict con chiavi 'vxp','vyp','vzp','vnormp'  shape (T, N)
    sfreq      : frequenza di campionamento in Hz
    percentile_clip : percentile per il clipping degli outlier
    save_path  : se fornito, salva la figura invece di mostrarla
    """
    vx = v['vxp']   # (T, N)
    vy = v['vyp']
    vz = v['vzp']
    vn = v['vnormp']

    T, N = vn.shape
    time = np.arange(T) / sfreq  # asse temporale in secondi

    # ── Clip outlier (velocità > percentile_clip) ────────────────────────────
    thresh = np.nanpercentile(vn, percentile_clip)
    mask   = vn < thresh          # (T, N) bool — True = campione valido

    # ── Media temporale sui campioni validi ──────────────────────────────────
    def masked_mean(arr):
        arr_m = np.where(mask, arr, np.nan)
        return np.nanmean(arr_m, axis=0)   # (N,)

    vx_mean = masked_mean(vx)
    vy_mean = masked_mean(vy)
    vz_mean = masked_mean(vz)
    vn_mean = masked_mean(vn)

    # Normalizza i vettori direzione per il quiver (lunghezza unitaria)
    norm = np.sqrt(vx_mean**2 + vy_mean**2 + vz_mean**2)
    norm[norm == 0] = 1
    ux = vx_mean / norm
    uy = vy_mean / norm
    uz = vz_mean / norm

    # ── Figura con 3 pannelli ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Phase Flow — Direzione media del flusso di fase EEG (banda alpha 8–13 Hz)',
                 fontsize=13, fontweight='bold')

    # ── Pannello 1: Vista 3D vettori flusso ──────────────────────────────────
    ax1 = fig.add_subplot(131, projection='3d')

    # Colore = velocità media (dopo clip)
    colors = cm.plasma(vn_mean / np.nanmax(vn_mean))

    ax1.quiver(loc[:,0], loc[:,1], loc[:,2],
               ux, uy, uz,
               length=12, normalize=False,
               color=colors, linewidth=1.5, arrow_length_ratio=0.3)

    ax1.scatter(loc[:,0], loc[:,1], loc[:,2],
                c=vn_mean, cmap='plasma', s=80, zorder=5)

    for i, ch in enumerate(CHANNELS):
        ax1.text(loc[i,0], loc[i,1], loc[i,2]+3, ch,
                 fontsize=6, ha='center')

    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title('Vista 3D — vettori unitari\n(colore = velocità media)')
    ax1.view_init(elev=25, azim=45)

    # ── Pannello 2: Proiezione 2D superiore (top view) ───────────────────────
    ax2 = fig.add_subplot(132)

    sc = ax2.scatter(loc[:,0], loc[:,1],
                     c=vn_mean, cmap='plasma', s=120, zorder=5,
                     vmin=0, vmax=np.nanmax(vn_mean))

    # Quiver proiettato sul piano XY
    ax2.quiver(loc[:,0], loc[:,1], ux, uy,
               scale=8, scale_units='inches',
               color='white', linewidth=1.2,
               width=0.005, headwidth=4,
               zorder=6)

    for i, ch in enumerate(CHANNELS):
        ax2.annotate(ch, (loc[i,0], loc[i,1]),
                     textcoords='offset points', xytext=(4,4),
                     fontsize=6, color='white')

    ax2.set_facecolor('#1a1a2e')
    plt.colorbar(sc, ax=ax2, label='Velocità media (mm/s)')
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)')
    ax2.set_title('Vista superiore (piano XY)\nfrecce = direzione flusso proiettata')
    ax2.set_aspect('equal')

    # Disegna cerchio testa
    theta = np.linspace(0, 2*np.pi, 100)
    r_head = HEAD_RADIUS_MM * np.sin(np.radians(90))
    ax2.plot(r_head*np.cos(theta), r_head*np.sin(theta),
             'w--', linewidth=0.8, alpha=0.4)

    # ── Pannello 3: vnormp nel tempo (media + std tra elettrodi) ─────────────
    ax3 = fig.add_subplot(133)

    # Media e std tra elettrodi, campioni validi
    vn_masked = np.where(mask, vn, np.nan)
    vn_t_mean = np.nanmean(vn_masked, axis=1)   # (T,)
    vn_t_std  = np.nanstd(vn_masked,  axis=1)

    # Sottocampiona per la visualizzazione (ogni 10 campioni = 20 Hz → 0.05s)
    step = max(1, T // 5000)
    t_plot  = time[::step]
    m_plot  = vn_t_mean[::step]
    s_plot  = vn_t_std[::step]

    ax3.fill_between(t_plot, m_plot - s_plot, m_plot + s_plot,
                     alpha=0.3, color='royalblue', label='±1 std')
    ax3.plot(t_plot, m_plot, color='royalblue', linewidth=0.8, label='Media')
    ax3.axhline(np.nanmedian(vn_t_mean), color='orange', linestyle='--',
                linewidth=1, label=f'Mediana {np.nanmedian(vn_t_mean):.0f}')

    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Velocità di fase (mm/s)')
    ax3.set_title(f'Velocità media tra elettrodi nel tempo\n(outlier > {percentile_clip}° pct rimossi)')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, time[-1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figura salvata in: {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Esecuzione standalone: riesegue la pipeline e poi visualizza
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    print("Rieseguo la pipeline per ottenere i risultati...")
    from main_phaseflow import run_pipeline

    results = run_pipeline(
        edf_path    = 'sub-001_ses-pre_task-rest_eeg.edf',
        fmin        = 8.0,
        fmax        = 13.0,
        speed_only  = False,
    )

    v = results['v']
    sfreq = results.get('sfreq', 200.0)

    print("\nGenerazione visualizzazione...")
    plot_phaseflow(v, sfreq=sfreq, save_path='phaseflow_visualization.png')
    print("Fatto!")
