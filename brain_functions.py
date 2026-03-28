"""
brain_functions.py  —  versione ottimizzata
============================================
Modifiche rispetto alla versione precedente:
  - phaseflow_cnem: loop su T timestep eliminato completamente.
    Il gradiente spaziale è ora calcolato con una singola operazione
    matriciale (B @ yphasor_matrix) su tutti i campioni in parallelo.
  - Aggiunto parametro 'chunk_size' per gestire la memoria su registrazioni lunghe.
"""

import numpy as np
from scipy.signal import hilbert
from scipy.spatial import ConvexHull

# --------------------------------------------------------------------------- #
# Import del modulo C-NEM compilato
# --------------------------------------------------------------------------- #
try:
    import cnem2d as _cnem
    _CNEM_AVAILABLE = True
except ImportError:
    _CNEM_AVAILABLE = False
    import warnings
    warnings.warn(
        "Modulo cnem2d non trovato. Assicurati che cnem2d.pyd/.so sia nel "
        "PYTHONPATH. Le funzioni grad_B_cnem e grad_cnem non saranno disponibili.",
        ImportWarning,
        stacklevel=2,
    )


# =========================================================================== #
#  1. phases_nodes                                                             #
# =========================================================================== #

def phases_nodes(yp: np.ndarray) -> np.ndarray:
    """
    Calcola la fase istantanea (unwrapped) in tutti i nodi.

    Parametri
    ----------
    yp : ndarray (T, N)

    Ritorna
    -------
    yphasep : ndarray (T, N)  — fasi in radianti, unwrapped nel tempo
    """
    print("calculating phases...", end="", flush=True)

    yp = np.asarray(yp, dtype=float)
    yp_centered = yp - yp.mean(axis=0)

    THRESHOLD = 100_000

    if yp.size < THRESHOLD:
        analytic = hilbert(yp_centered, axis=0)
        yphasep  = np.unwrap(np.angle(analytic), axis=0)
    else:
        T, N    = yp_centered.shape
        yphasep = np.zeros_like(yp_centered)
        for jj in range(N):
            y = yp_centered[:, jj]
            yphasep[:, jj] = np.unwrap(np.angle(hilbert(y)))

    print(" done")
    return yphasep


# =========================================================================== #
#  2. grad_B_cnem                                                              #
# =========================================================================== #

def grad_B_cnem(xyz: np.ndarray, boundary_facets=None) -> np.ndarray:
    """
    USANDO CNEM 
    Calcola la matrice B per il gradiente ∇V su punti 3-D sparsi.

    Parametri
    ----------
    xyz : ndarray (N, 3)
    boundary_facets : ndarray (M,2) o (M,3), opzionale

    Ritorna
    -------
    B : ndarray (3*N, N)
    
    if not _CNEM_AVAILABLE:
        raise RuntimeError("Modulo cnem2d non disponibile.")

    from cnem_functions import _prepare_cnem2d_inputs, _parse_scni_output

    xyz = np.asarray(xyz, dtype=float)

    bdy_segs = None
    if boundary_facets is not None:
        bf = np.asarray(boundary_facets, dtype=int)
        if bf.ndim == 2 and bf.shape[1] == 3:
            bdy_segs = _triangles_to_boundary_segments(bf)
        elif bf.ndim == 2 and bf.shape[1] == 2:
            bdy_segs = bf

    xy_flat, nb_front, ind_front, pca_axes, _ = _prepare_cnem2d_inputs(xyz, bdy_segs)
    result = _cnem.SCNI_CNEM2D(xy_flat, nb_front, ind_front)

    N = xyz.shape[0]
    B, _, _ = _parse_scni_output(result, N, pca_axes)
    return B


def _triangles_to_boundary_segments(triangles: np.ndarray):
    from collections import Counter
    edge_count = Counter()
    for tri in triangles:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge = (min(a, b), max(a, b))
            edge_count[edge] += 1
    bdy = [list(e) for e, cnt in edge_count.items() if cnt == 1]
    return np.array(bdy, dtype=int) if bdy else None
"""

    """
    C-NEM calcola il gradiente solo per 3 elettrodi su 19 perché nb_contrib è quasi tutto zero.
    La soluzione più pragmatica per il porting è bypassare C-NEM per il calcolo del gradiente spaziale 
    e usare invece un metodo di interpolazione standard 
    (RBF — Radial Basis Functions) che è più robusto con dati sparsi su sfera e dà risultati corretti per tutti i (esempio: 19) elettrodi.
    
    
    Calcola la matrice B per il gradiente ∇V su punti 3-D sparsi.
    Usa interpolazione RBF (thin-plate spline) sulla proiezione PCA 2D.
    """
    xyz = np.asarray(xyz, dtype=float)
    N = xyz.shape[0]

    # PCA 3D → 2D (stessa proiezione di cnem_functions)
    center = xyz.mean(axis=0)
    xc = xyz - center
    _, _, Vt = np.linalg.svd(xc, full_matrices=False)
    pca_axes = Vt[:2].T   # (3, 2)
    xy2d = xc @ pca_axes  # (N, 2)

    # Matrice gradiente RBF thin-plate spline
    # Per ogni nodo i, B[d*N+i, j] = peso della funzione base j
    # nel calcolo di ∂V/∂x_d valutata nel nodo i
    from scipy.interpolate import RBFInterpolator

    B_2d = np.zeros((2 * N, N))
    eps = 1e-8

    for j in range(N):
        # Funzione base j: vettore canonico e_j
        ej = np.zeros(N)
        ej[j] = 1.0
        rbf = RBFInterpolator(xy2d, ej, kernel='thin_plate_spline')
        # Gradiente numerico ∂/∂x e ∂/∂y
        dx = np.zeros_like(xy2d); dx[:, 0] = eps
        dy = np.zeros_like(xy2d); dy[:, 1] = eps
        gx = (rbf(xy2d + dx) - rbf(xy2d - dx)) / (2 * eps)
        gy = (rbf(xy2d + dy) - rbf(xy2d - dy)) / (2 * eps)
        B_2d[0*N:1*N, j] = gx
        B_2d[1*N:2*N, j] = gy

    # Riproiezione 2D → 3D tramite PCA
    B_3d = np.zeros((3 * N, N))
    for d in range(3):
        B_3d[d*N:(d+1)*N, :] = (
            pca_axes[d, 0] * B_2d[0*N:1*N, :]
            + pca_axes[d, 1] * B_2d[1*N:2*N, :]
        )
    return B_3d
# =========================================================================== #
#  3. grad_cnem                                                                #
# =========================================================================== #

def grad_cnem(xyz_or_B, V: np.ndarray, boundary_facets=None) -> np.ndarray:
    """
    Gradiente ∇V valutato su punti 3-D sparsi.

    Accetta sia coordinate (N,3) sia la matrice B precalcolata (3N,N).
    V può essere reale o complesso, shape (N,) o (N, T) per elaborazione batch.

    Ritorna
    -------
    gradV : ndarray (N, 3)  se V è (N,)
            ndarray (T, N, 3) se V è (N, T)   ← batch mode
    """
    xyz_or_B = np.asarray(xyz_or_B)
    V = np.asarray(V)
    is_batch = V.ndim == 2   # (N, T)

    if is_batch:
        N = V.shape[0]
    else:
        N = len(V.ravel())
        V = V.ravel()

    # Scegli/costruisci B
    if xyz_or_B.ndim == 2 and xyz_or_B.shape[1] == 3 and xyz_or_B.shape[0] == N:
        B = grad_B_cnem(xyz_or_B, boundary_facets)
    else:
        B = xyz_or_B   # già la matrice B

    # Prodotto matriciale
    if np.iscomplexobj(V):
        G_re = B @ V.real    # (3N, T) o (3N,)
        G_im = B @ V.imag
        G    = G_re + 1j * G_im
    else:
        G = B @ V            # (3N, T) o (3N,)

    if is_batch:
        T = V.shape[1]
        # G ha shape (3N, T) → vogliamo (T, N, 3)
        # Reshape: (3, N, T) poi transpose → (T, N, 3)
        grad_V = G.reshape(3, N, T).transpose(2, 1, 0)   # (T, N, 3)
    else:
        grad_V = G.reshape(3, N).T   # (N, 3)

    return grad_V


# =========================================================================== #
#  4. phaseflow_cnem  —  versione vettorizzata (nessun loop su T)             #
# =========================================================================== #

def phaseflow_cnem(yphasep: np.ndarray,
                   loc: np.ndarray,
                   dt: float,
                   speedonlyflag: bool = False,
                   chunk_size: int = 5000) -> dict:
    """
    Calcola il flusso di fase istantaneo in ogni punto temporale.

    Versione completamente vettorizzata: nessun loop Python su T.
    Usa chunk_size per evitare picchi di memoria su registrazioni lunghe.

    Parametri
    ----------
    yphasep    : ndarray (T, N)  — fasi unwrapped
    loc        : ndarray (N, 3)  — coordinate mm
    dt         : float           — passo temporale (s)
    speedonlyflag : bool         — True → solo vnormp
    chunk_size : int             — campioni per chunk (default 5000)
                                   Riduci se memoria insufficiente.

    Ritorna
    -------
    v : dict
        'vnormp' ndarray (T, N)
        'vxp'    ndarray (T, N)  (se speedonlyflag=False)
        'vyp'    ndarray (T, N)
        'vzp'    ndarray (T, N)
    """
    yphasep = np.asarray(yphasep, dtype=float)
    loc     = np.asarray(loc,     dtype=float)
    T, N    = yphasep.shape

    # ------------------------------------------------------------------ #
    # ∂φ/∂t  — una sola chiamata numpy su tutto il tensore               #
    # ------------------------------------------------------------------ #
    dphidtp = np.gradient(yphasep, dt, axis=0)   # (T, N)
    print("dphidt done...", end="", flush=True)

    # ------------------------------------------------------------------ #
    # Matrice B (calcolata UNA sola volta, boundary gestito internamente) #
    # ------------------------------------------------------------------ #
    # Non passiamo boundary_facets: _prepare_cnem2d_inputs calcola
    # autonomamente il contorno convesso sui punti 2D proiettati via PCA.
    # Passare hull.simplices 3D causerebbe un IndexError nel C++ perché
    # i simplici 3D (triangoli) non sono compatibili con il contorno 2D.
    B = grad_B_cnem(loc)   # (3N, N) — costante per tutto il segnale
    print("B built...", end="", flush=True)

    # ------------------------------------------------------------------ #
    # Gradiente spaziale — vettorizzato con chunking                     #
    # ------------------------------------------------------------------ #
    # Strategia:
    #   yphasor = exp(i·φ)  →  shape (T, N)
    #   Per ogni chunk: B @ yphasor[chunk].T  →  (3N, chunk_len)
    #   poi ∇φ = Re(-i · grad_phasor · conj(phasor))
    #
    # Dimensioni intermedie per chunk_size=5000, N=19:
    #   yphasor chunk : (5000, 19) complex → ~1.5 MB
    #   B @ chunk     : (57,   19) @ (19, 5000) = (57, 5000) → ~2.3 MB
    #   totale ~5 MB per chunk — molto gestibile

    dphidxp = np.zeros((T, N))
    dphidyp = np.zeros((T, N))
    dphidzp = np.zeros((T, N))

    n_chunks = int(np.ceil(T / chunk_size))
    for c in range(n_chunks):
        t0 = c * chunk_size
        t1 = min(t0 + chunk_size, T)

        phi_chunk    = yphasep[t0:t1, :]          # (chunk, N)
        phasor_chunk = np.exp(1j * phi_chunk)     # (chunk, N)

        # B @ phasor.T → (3N, chunk),  poi reshape → (3, N, chunk)
        # trasponiamo phasor per avere colonne=campioni
        G = B @ phasor_chunk.T                    # (3N, chunk) complex
        G3 = G.reshape(3, N, -1)                  # (3, N, chunk)

        # conj(phasor).T ha shape (N, chunk)
        conj_ph = np.conj(phasor_chunk).T         # (N, chunk)

        # ∇φ = Re(-i · G · conj(z))  — moltiplicazione element-wise su asse N
        # G3[d] ha shape (N, chunk),  conj_ph ha shape (N, chunk)
        dphidxp[t0:t1, :] = np.real(-1j * G3[0] * conj_ph).T   # (chunk, N)
        dphidyp[t0:t1, :] = np.real(-1j * G3[1] * conj_ph).T
        dphidzp[t0:t1, :] = np.real(-1j * G3[2] * conj_ph).T

        if (c + 1) % 5 == 0 or c == n_chunks - 1:
            print(f"\r  gradphi chunk {c+1}/{n_chunks} ({t1}/{T} campioni)...",
                  end="", flush=True)

    print("  gradphi done...", flush=True)

    # ------------------------------------------------------------------ #
    # Velocità                                                            #
    # ------------------------------------------------------------------ #
    normgradphi = np.sqrt(dphidxp**2 + dphidyp**2 + dphidzp**2)

    # Evita divisioni per zero (punti con gradiente nullo)
    with np.errstate(divide="ignore", invalid="ignore"):
        vnormp = np.where(normgradphi > 0,
                          np.abs(dphidtp) / normgradphi,
                          np.nan)

    v = {"vnormp": vnormp}

    if not speedonlyflag:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_norm = np.where(normgradphi > 0, 1.0 / normgradphi, np.nan)
        v["vxp"] = vnormp * (-dphidxp * inv_norm)
        v["vyp"] = vnormp * (-dphidyp * inv_norm)
        v["vzp"] = vnormp * (-dphidzp * inv_norm)

    print("v done...")
    return v
