import numpy as np
import cnem2d
from scipy.spatial import ConvexHull
from collections import defaultdict

print("file:", cnem2d.__file__)

# ── 19 elettrodi EEG reali ─────────────────────────────────────────────────
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
N = len(channels)

# PCA 3D → 2D
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

xy_flat = tuple(xy2d.ravel(order='C').tolist())
nb_front = (len(path),)
ind_front = tuple(path)

print(f"\nTest 19 elettrodi EEG:")
print(f"  N={N}, boundary={len(path)} nodi, interni={N-len(path)}")
print(f"  ind_front={ind_front}")

try:
    result = cnem2d.SCNI_CNEM2D(xy_flat, nb_front, ind_front)
    print(f"\n  STATUS = OK!")
    print(f"  out[0] new_old = {result[0]}")
    print(f"  out[1] old_new = {result[1]}")
    print(f"  out[4] nb_contrib = {result[4]}")
    print(f"  out[5] INV len = {len(result[5])}")
    print(f"  out[6] Grad len = {len(result[6])}")
except Exception as e:
    print(f"\n  STATUS = EXCEPTION: {type(e).__name__}: {e}")

print("\ndone")
