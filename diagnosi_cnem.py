"""
diagnosi_cnem.py  — diagnostica della matrice B e della proiezione PCA
"""
import numpy as np
from scipy.spatial import ConvexHull
from cnem_functions import _prepare_cnem2d_inputs, _parse_scni_output
import cnem2d as _cnem

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

loc = np.array([sph_to_cart(*_COORDS[ch]) for ch in CHANNELS])

# ── Proiezione PCA ──────────────────────────────────────────────────────────
center = loc.mean(axis=0)
xc = loc - center
_, _, Vt = np.linalg.svd(xc, full_matrices=False)
pca_axes = Vt[:2].T
xy2d = xc @ pca_axes

print("=== Proiezione PCA 2D ===")
print(f"Assi PCA: \n{pca_axes}")
print(f"\nCoordinate 2D elettrodi:")
for i, ch in enumerate(CHANNELS):
    print(f"  {ch:4s}: x2d={xy2d[i,0]:7.1f}  y2d={xy2d[i,1]:7.1f}")

# ── Risultato SCNI ──────────────────────────────────────────────────────────
xy_flat, nb_front, ind_front, pca_axes_out, _ = _prepare_cnem2d_inputs(loc)
result = _cnem.SCNI_CNEM2D(xy_flat, nb_front, ind_front)

N = len(CHANNELS)
new_old    = np.array(result[0], dtype=int) - 1
nb_contrib = np.array(result[4], dtype=int)
inv        = np.array(result[5], dtype=int) - 1
grad_flat  = np.array(result[6], dtype=float)

new_old = np.clip(new_old, 0, N-1)
inv     = np.clip(inv,     0, N-1)

print(f"\n=== Output SCNI ===")
print(f"new_old (0-based): {new_old}")
print(f"nb_contrib: {nb_contrib}  (sum={nb_contrib.sum()})")
print(f"INV len: {len(inv)}")
print(f"Grad len: {len(grad_flat)}")

# ── Matrice B ───────────────────────────────────────────────────────────────
B, _, _ = _parse_scni_output(result, N, pca_axes_out)
print(f"\n=== Matrice B ===")
print(f"Shape: {B.shape}")
print(f"Righe non-zero: {np.sum(np.any(B != 0, axis=1))} / {B.shape[0]}")
print(f"Colonne non-zero: {np.sum(np.any(B != 0, axis=0))} / {B.shape[1]}")
print(f"Norma di Frobenius: {np.linalg.norm(B):.4f}")

# Mostra quali elettrodi hanno riga non-zero in B
print(f"\nElettrodI con gradiente calcolato (righe non-zero in Bx, By, Bz):")
for d, dim in enumerate(['x','y','z']):
    nonzero = np.where(np.any(B[d*N:(d+1)*N, :] != 0, axis=1))[0]
    print(f"  B{dim}: elettrodi {[CHANNELS[i] for i in nonzero]}")
