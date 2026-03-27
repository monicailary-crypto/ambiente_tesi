"""
test_cnem2d.py
==============
Script diagnostico — esegui PRIMA di main_phaseflow.py.
Testa cnem2d con input progressivamente più complessi
per trovare esattamente dove crasha.

Uso:
    python test_cnem2d.py
"""

import numpy as np
import cnem2d

def run_test(name, xy_flat, nb_front, ind_front):
    print(f"\n{'─'*50}")
    print(f"TEST: {name}")
    print(f"  N         = {len(xy_flat)//2}")
    print(f"  nb_front  = {nb_front}  (sum={sum(nb_front)})")
    print(f"  ind_front = {ind_front}")
    try:
        result = cnem2d.SCNI_CNEM2D(xy_flat, nb_front, ind_front)
        print(f"  STATUS    = OK")
        print(f"  out[0] new_old (raw) = {result[0]}")
        print(f"  out[1] old_new (raw) = {result[1]}")
        print(f"  out[4] nb_contrib    = {result[4]}")
        print(f"  out[5] INV len       = {len(result[5])}")
        print(f"  out[6] Grad len      = {len(result[6])}")
        return result
    except Exception as e:
        print(f"  STATUS    = CRASH: {type(e).__name__}: {e}")
        return None

# ─────────────────────────────────────────────────────────
# TEST 1: Triangolo (3 punti, tutti sul bordo) — caso minimo
# ─────────────────────────────────────────────────────────
run_test(
    "Triangolo 3pt (indici 0-based)",
    xy_flat   = (0.0, 0.0,  1.0, 0.0,  0.5, 0.866),
    nb_front  = (3,),
    ind_front = (0, 1, 2),
)

# ─────────────────────────────────────────────────────────
# TEST 2: Triangolo con indici 1-based (per verificare convenzione)
# ─────────────────────────────────────────────────────────
run_test(
    "Triangolo 3pt (indici 1-based)",
    xy_flat   = (0.0, 0.0,  1.0, 0.0,  0.5, 0.866),
    nb_front  = (3,),
    ind_front = (1, 2, 3),   # 1-based
)

# ─────────────────────────────────────────────────────────
# TEST 3: Quadrato con punto interno (4 bordo + 1 interno)
# ─────────────────────────────────────────────────────────
run_test(
    "Quadrato 4pt bordo + 1 interno (0-based CCW)",
    xy_flat   = (0.0,0.0,  1.0,0.0,  1.0,1.0,  0.0,1.0,  0.5,0.5),
    nb_front  = (4,),
    ind_front = (0, 1, 2, 3),
)

# ─────────────────────────────────────────────────────────
# TEST 4: Quadrato 1-based
# ─────────────────────────────────────────────────────────
run_test(
    "Quadrato 4pt bordo + 1 interno (1-based CCW)",
    xy_flat   = (0.0,0.0,  1.0,0.0,  1.0,1.0,  0.0,1.0,  0.5,0.5),
    nb_front  = (4,),
    ind_front = (1, 2, 3, 4),
)

# ─────────────────────────────────────────────────────────
# TEST 5: Cerchio 8pt (geometria simile a scalpo EEG)
# ─────────────────────────────────────────────────────────
angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 punti sul bordo
pts_circle = np.column_stack([np.cos(angles), np.sin(angles)])
pts_with_center = np.vstack([pts_circle, [0.0, 0.0]])  # + centro
xy_circle = tuple(pts_with_center.ravel().tolist())

run_test(
    "Cerchio 8pt bordo + 1 centro (0-based CCW)",
    xy_flat   = xy_circle,
    nb_front  = (8,),
    ind_front = tuple(range(8)),
)

# ─────────────────────────────────────────────────────────
# TEST 6: 19 elettrodi EEG reali (input della pipeline)
# ─────────────────────────────────────────────────────────
HEAD_RADIUS_MM = 85.0
_COORDS = {
    "Fp1":(-18,72),"Fp2":(18,72),"F7":(-54,72),"F3":(-36,54),"FZ":(0,54),
    "T3":(-90,90),"C3":(-54,90),"CZ":(0,90),"C4":(54,90),"T4":(90,90),
    "T5":(-54,126),"P3":(-36,126),"PZ":(0,126),"P4":(36,126),"T6":(54,126),
    "O1":(-18,144),"O2":(18,144),"F4":(36,54),"F8":(54,72),
}
channels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']

def sph_to_cart(th_deg, ph_deg, r=HEAD_RADIUS_MM):
    th, ph = np.radians(th_deg), np.radians(ph_deg)
    return np.array([r*np.sin(ph)*np.cos(th), r*np.sin(ph)*np.sin(th), r*np.cos(ph)])

loc = np.array([sph_to_cart(*_COORDS[ch]) for ch in channels])

# PCA 3D → 2D
from scipy.spatial import ConvexHull
from collections import defaultdict
center = loc.mean(axis=0)
xc = loc - center
_, _, Vt = np.linalg.svd(xc, full_matrices=False)
pca_axes = Vt[:2].T
xy2d = xc @ pca_axes

hull = ConvexHull(xy2d)
adj_hull = defaultdict(set)
for s in hull.simplices:
    adj_hull[int(s[0])].add(int(s[1]))
    adj_hull[int(s[1])].add(int(s[0]))

start = int(hull.simplices[0, 0])
path = [start]; visited = {start}; current = start
for _ in range(len(hull.vertices)-1):
    for nxt in sorted(adj_hull[current]):
        if nxt not in visited:
            path.append(nxt); visited.add(nxt); current = nxt; break

pts = xy2d[path]; n_b = len(pts)
area = 0.5*sum(pts[i,0]*pts[(i+1)%n_b,1]-pts[(i+1)%n_b,0]*pts[i,1] for i in range(n_b))
if area < 0: path = list(reversed(path))

xy_flat_eeg = tuple(xy2d.ravel(order='C').tolist())

# Test con 0-based
run_test(
    "19 elettrodi EEG (0-based CCW)",
    xy_flat   = xy_flat_eeg,
    nb_front  = (len(path),),
    ind_front = tuple(path),
)

# Test con 1-based
run_test(
    "19 elettrodi EEG (1-based CCW)",
    xy_flat   = xy_flat_eeg,
    nb_front  = (len(path),),
    ind_front = tuple(i+1 for i in path),
)

print(f"\n{'='*50}")
print("Fine test diagnostici.")
print("Incolla l'output completo in chat.")
