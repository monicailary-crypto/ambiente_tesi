"""
topoplot_phaseflow.py  — versione corretta
==========================================
Usa le coordinate PCA 2D (già corrette) invece della proiezione azimutale.
Tutte le coordinate sono normalizzate su [-1, 1] rispetto al raggio massimo.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

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

loc = np.array([sph_to_cart(*_COORDS[ch]) for ch in CHANNELS])  # (19,3)

def get_xy2d(loc):
    """Proiezione PCA 3D->2D, normalizzata su [-1,1]."""
    center = loc.mean(axis=0)
    xc = loc - center
    _, _, Vt = np.linalg.svd(xc, full_matrices=False)
    pca_axes = Vt[:2].T          # (3,2)
    xy = xc @ pca_axes           # (N,2) in mm
    r_max = HEAD_RADIUS_MM
    xy_norm = xy / r_max         # (N,2) in [-1,1]
    return xy_norm, pca_axes

def draw_head(ax, lw=2.5):
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=lw, zorder=4)
    # Naso
    ax.plot([-0.1, 0, 0.1], [0.99, 1.12, 0.99], 'k-', lw=lw, zorder=4)
    # Orecchie
    ear_th = np.linspace(np.pi*0.45, np.pi*1.55, 60)
    ex = np.cos(ear_th + np.pi/2)
    ey = np.sin(ear_th + np.pi/2)
    ax.plot(-ex - 0.005, ey, 'k-', lw=lw, zorder=4)
    ax.plot( ex + 0.005, ey, 'k-', lw=lw, zorder=4)

def topoplot_flow(v, sfreq=200.0, percentile_clip=95.0,
                  save_path='topoplot_phaseflow.png'):

    xy2d, pca_axes = get_xy2d(loc)   # (N,2) normalizzate

    vx = v['vxp']
    vy = v['vyp']
    vz = v['vzp']
    vn = v['vnormp']
    T, N = vn.shape

    thresh = np.nanpercentile(vn, percentile_clip)
    mask   = vn < thresh

    def mmean(arr):
        return np.nanmean(np.where(mask, arr, np.nan), axis=0)

    vx_m = mmean(vx)
    vy_m = mmean(vy)
    vz_m = mmean(vz)
    vn_m = mmean(vn)

    # Proiezione vettori 3D -> 2D tramite PCA axes
    v3d  = np.column_stack([vx_m, vy_m, vz_m])  # (N,3)
    v2d  = v3d @ pca_axes                        # (N,2) in mm/s

    # Normalizza frecce: max = 15% del raggio
    mag2d = np.sqrt(v2d[:,0]**2 + v2d[:,1]**2)
    max_mag = mag2d.max()
    v2d_norm = v2d / max_mag * 0.15 if max_mag > 0 else v2d

    # Heatmap
    grid_res = 300
    xi = np.linspace(-1.0, 1.0, grid_res)
    yi = np.linspace(-1.0, 1.0, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    rbf = RBFInterpolator(xy2d, vn_m, kernel='thin_plate_spline', smoothing=0.5)
    Zi  = rbf(np.column_stack([Xi.ravel(), Yi.ravel()])).reshape(grid_res, grid_res)

    outside = Xi**2 + Yi**2 > 1.0
    Zi[outside] = np.nan

    fig, ax = plt.subplots(figsize=(8, 9))

    im = ax.contourf(Xi, Yi, Zi, levels=60, cmap='RdYlBu_r', zorder=1)
    ax.contour(Xi, Yi, Zi, levels=12, colors='white',
               linewidths=0.4, alpha=0.35, zorder=2)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label('Velocita di fase media (mm/s)', fontsize=11)

    draw_head(ax)

    ax.scatter(xy2d[:,0], xy2d[:,1],
               c=vn_m, cmap='RdYlBu_r', s=100,
               edgecolors='k', linewidths=0.8, zorder=5,
               vmin=np.nanmin(Zi[~outside]), vmax=np.nanmax(Zi[~outside]))

    for i, ch in enumerate(CHANNELS):
        ax.annotate(ch, (xy2d[i,0], xy2d[i,1]),
                    xytext=(5,5), textcoords='offset points',
                    fontsize=7.5, fontweight='bold', zorder=6)

    ax.quiver(xy2d[:,0], xy2d[:,1],
              v2d_norm[:,0], v2d_norm[:,1],
              scale=1.0, scale_units='xy',
              color='black', width=0.005,
              headwidth=4, headlength=5, zorder=7)

    ax.text( 0.0,  1.18, 'Anteriore (Naso)', ha='center', va='bottom', fontsize=9, style='italic')
    ax.text( 0.0, -1.18, 'Posteriore',       ha='center', va='top',    fontsize=9, style='italic')
    ax.text(-1.18,  0.0, 'Sinistro',         ha='right',  va='center', fontsize=9, style='italic')
    ax.text( 1.18,  0.0, 'Destro',           ha='left',   va='center', fontsize=9, style='italic')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    med = np.nanmedian(vn_m)
    ax.set_title(
        f'Flusso di fase EEG - banda alpha (8-13 Hz)\n'
        f'Velocita mediana: {med:.0f} mm/s  '
        f'[outlier >{percentile_clip} pct rimossi]\n'
        'Frecce = direzione media del flusso proiettata sullo scalpo',
        fontsize=11, pad=12
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Salvato: {save_path}")
    plt.show()
    return fig


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from main_phaseflow import run_pipeline

    print("Eseguo pipeline...")
    results = run_pipeline(
        edf_path   = 'sub-001_ses-pre_task-rest_eeg.edf',
        fmin       = 8.0,
        fmax       = 13.0,
        speed_only = False,
    )
    v     = results['v']
    sfreq = results.get('sfreq', 200.0)

    print("\nGenerazione topoplot...")
    topoplot_flow(v, sfreq=sfreq, save_path='topoplot_phaseflow.png')
    print("Fatto!")
