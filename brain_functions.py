import numpy as np
from scipy.signal import hilbert
from scipy.spatial import ConvexHull
from cnem_functions import _prepare_cnem2d_inputs, _parse_scni_output, _order_boundary_segments

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

    Equivalente MATLAB:
        yphasep = unwrap(angle(hilbert(bsxfun(@minus, yp, mean(yp)))));

    Parametri
    ----------
    yp : ndarray (T, N)
        Segnali: T campioni temporali, N canali/regioni.

    Ritorna
    -------
    yphasep : ndarray (T, N)
        Fasi istantanee in radianti (unwrapped nel tempo).
    """
    print("calculating phases...", end="", flush=True)

    yp = np.asarray(yp, dtype=float)
    yp_centered = yp - yp.mean(axis=0)    # bsxfun(@minus, yp, mean(yp))

    THRESHOLD = 100_000

    if yp.size < THRESHOLD:
        # versione vettorizzata (come il ramo "short" in MATLAB)
        analytic  = hilbert(yp_centered, axis=0)
        yphasep   = np.unwrap(np.angle(analytic), axis=0)
    else:
        # versione per-canale (come il ramo "else" in MATLAB)
        T, N = yp_centered.shape
        yphasep = np.zeros_like(yp_centered)
        for jj in range(N):
            y = yp_centered[:, jj]
            yphasep[:, jj] = np.unwrap(np.angle(hilbert(y)))

    print(" done")
    return yphasep


# =========================================================================== #
#  2. grad_B_cnem                                                              #
# =========================================================================== #

def grad_B_cnem(xyz: np.ndarray, boundary_facets=None):
    """
    Calcola la matrice B per il gradiente ∇V su punti 3-D sparsi.

    Equivalente MATLAB:  B = grad_B_cnem(XYZ, [IN_Tri_Ini])
    Uso:  gradV = B @ V   oppure   gradV = grad_cnem(B, V)

    Parametri
    ----------
    xyz : ndarray (N, 3)  — coordinate 3-D dei punti
    boundary_facets : ndarray (M, 2) o (M, 3), opzionale
        Triangolazione/segmenti della superficie di bordo.
        Se None viene usata la ConvexHull.

    Ritorna
    -------
    B : ndarray (3*N, N)
        Matrice operatore gradiente.
        Grad_V = B @ V  → vettore (3*N,)
        reshape(Grad_V, (N, 3)) → [∂V/∂x, ∂V/∂y, ∂V/∂z] per nodo.

        (Nota: in MATLAB il reshape era (4, N).T con 4a colonna = V;
         qui usiamo (3, N).T — solo le 3 componenti del gradiente.)
    """
    if not _CNEM_AVAILABLE:
        raise RuntimeError("Modulo cnem2d non disponibile.")

    xyz = np.asarray(xyz, dtype=float)

    # Gestione boundary_facets: converti triangoli → segmenti di bordo 2-D
    # (cnem2d lavora con contorni, non triangolazioni di superficie)
    bdy_segs = None
    if boundary_facets is not None:
        bf = np.asarray(boundary_facets, dtype=int)
        if bf.ndim == 2 and bf.shape[1] == 3:
            # Triangolazione → bordo = lati che appaiono una sola volta
            bdy_segs = _triangles_to_boundary_segments(bf)
        elif bf.ndim == 2 and bf.shape[1] == 2:
            bdy_segs = bf

    xy_flat, nb_front, ind_front, pca_axes, _ = _prepare_cnem2d_inputs(xyz, bdy_segs)

    print(f"Sto passando al C++ {len(xy_flat)} punti.")
    result = _cnem.SCNI_CNEM2D(xy_flat, nb_front, ind_front)

    N = xyz.shape[0]
    B, _, _ = _parse_scni_output(result, N, pca_axes)
    return B


def _triangles_to_boundary_segments(triangles: np.ndarray) -> np.ndarray:
    """
    Dato un array (M, 3) di triangoli, restituisce i segmenti di bordo
    (lati che appaiono esattamente una volta).
    """
    from collections import Counter
    edge_count = Counter()
    for tri in triangles:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge = (min(a, b), max(a, b))
            edge_count[edge] += 1
    bdy = [list(e) for e, cnt in edge_count.items() if cnt == 1]
    return np.array(bdy, dtype=int) if bdy else None


# =========================================================================== #
#  3. grad_cnem                                                                #
# =========================================================================== #

def grad_cnem(xyz_or_B, V: np.ndarray, boundary_facets=None) -> np.ndarray:
    """
    Gradiente ∇V valutato su punti 3-D sparsi.

    Equivalente MATLAB:
        gradV = grad_cnem(XYZ, V, [IN_Tri_Ini])
        gradV = grad_cnem(B, V)        % B precalcolata

    Parametri
    ----------
    xyz_or_B : ndarray (N, 3) oppure ndarray (3*N, N)
        Se shape (N, 3): coordinate 3-D → calcola B internamente.
        Se shape (3*N, N): usa la matrice B precalcolata.
    V : ndarray (N,) — reale o complesso
        Valore della funzione scalare nei nodi.
    boundary_facets : opzionale, passato a grad_B_cnem se xyz_or_B è coord.

    Ritorna
    -------
    gradV : ndarray (N, 3)
        [∂V/∂x, ∂V/∂y, ∂V/∂z] per ogni nodo.
    """
    xyz_or_B = np.asarray(xyz_or_B)
    V = np.asarray(V).ravel()
    N = len(V)

    # Distingui matrice B (quadrata/rettangolare con 3N righe) da matrice coord
    if xyz_or_B.ndim == 2 and xyz_or_B.shape[1] == 3 and xyz_or_B.shape[0] == N:
        # È una matrice di coordinate (N, 3)
        B = grad_B_cnem(xyz_or_B, boundary_facets)
    else:
        # È già la matrice B
        B = xyz_or_B

    if np.iscomplexobj(V):
        # Separa reale e immaginaria per rispettare la linearità di B
        Grad_V_re = B @ V.real
        Grad_V_im = B @ V.imag
        Grad_V = Grad_V_re + 1j * Grad_V_im
    else:
        Grad_V = B @ V

    # reshape: (3*N,) → (N, 3)  [equivalente a reshape(Grad_V, 3, [])' in MATLAB]
    grad_V_mat = Grad_V.reshape(3, N).T   # (N, 3): colonne = x, y, z
    return grad_V_mat


# =========================================================================== #
#  4. phaseflow_cnem                                                           #
# =========================================================================== #

def phaseflow_cnem(yphasep: np.ndarray,
                   loc: np.ndarray,
                   dt: float,
                   speedonlyflag: bool = False) -> dict:
    """
    Calcola il flusso di fase istantaneo in ogni punto temporale.

    Equivalente MATLAB:  v = phaseflow_cnem(yphasep, loc, dt, speedonlyflag)

    Parametri
    ----------
    yphasep : ndarray (T, N)
        Matrice delle fasi (unwrapped), T campioni × N canali.
        Si assume già unwrapped nel tempo (usa phases_nodes() per ottenerla).
    loc : ndarray (N, 3)
        Coordinate 3-D delle regioni/elettrodi.
    dt : float
        Passo temporale in secondi.
    speedonlyflag : bool, default False
        True  → calcola solo la velocità scalare vnormp (più rapido).
        False → calcola anche le componenti vettoriali vxp, vyp, vzp.

    Ritorna
    -------
    v : dict con chiavi:
        'vnormp' ndarray (T, N) — velocità scalare (m/s)
        'vxp'    ndarray (T, N) — componente x  (solo se speedonlyflag=False)
        'vyp'    ndarray (T, N) — componente y
        'vzp'    ndarray (T, N) — componente z

    Note matematiche
    ----------------
    Rubino et al. (2006):   |v| = |∂φ/∂t| / ‖∇φ‖
    Direzione:               v⃗  = |v| · (−∇φ / ‖∇φ‖)

    Il gradiente spaziale è calcolato sul fasore z = e^{iφ} per evitare
    discontinuità di wrap-around, poi riconvertito:
        ∇φ = Re(−i · ∇z · conj(z))
    """
    yphasep = np.asarray(yphasep, dtype=float)
    loc     = np.asarray(loc,     dtype=float)
    T, N    = yphasep.shape

    # ------------------------------------------------------------------ #
    # ∂φ/∂t  — derivata temporale della fase                             #
    # ------------------------------------------------------------------ #
    # MATLAB: [~, dphidtp] = gradient(yphasep, dt)
    # gradient() su matrice 2-D restituisce [d/dcols, d/drows].
    # Il secondo output è la derivata lungo le righe (=tempo).
    # np.gradient con spacing=(1, dt) su assi (0=tempo, 1=spazio):
    dphidtp, _ = np.gradient(yphasep, dt, 1, axis=(0, 1))
    # Alternativa equivalente e più esplicita:
    # dphidtp = np.gradient(yphasep, dt, axis=0)
    print("dphidt done...", end="", flush=True)

    # ------------------------------------------------------------------ #
    # Superficie di bordo (alpha shape → ConvexHull come approssimazione) #
    # ------------------------------------------------------------------ #
    try:
        hull = ConvexHull(loc)
        boundary_facets = hull.simplices    # (M, 3) triangoli
    except Exception:
        boundary_facets = None

    # ------------------------------------------------------------------ #
    # Matrice B del gradiente (calcolata una sola volta)                  #
    # ------------------------------------------------------------------ #
    B = grad_B_cnem(loc, boundary_facets)

    # ------------------------------------------------------------------ #
    # Gradiente spaziale della fase via fasori                            #
    # ------------------------------------------------------------------ #
    dphidxp = np.zeros((T, N))
    dphidyp = np.zeros((T, N))
    dphidzp = np.zeros((T, N))

    for j in range(T):
        yphase  = yphasep[j, :]           # (N,) — fase al tempo j
        yphasor = np.exp(1j * yphase)     # z = e^{iφ}, proiezione sul cerchio unitario

        # Gradiente del fasore nello spazio 3-D → shape (N, 3)
        gradphasor = grad_cnem(B, yphasor)

        # ∇φ = Re(−i · ∇z · conj(z))   [perché |z|=1 ⟹ 1/z = conj(z)]
        conj_ph = np.conj(yphasor)
        dphidxp[j, :] = np.real(-1j * gradphasor[:, 0] * conj_ph)
        dphidyp[j, :] = np.real(-1j * gradphasor[:, 1] * conj_ph)
        dphidzp[j, :] = np.real(-1j * gradphasor[:, 2] * conj_ph)

    print("gradphi done...", end="", flush=True)

    # ------------------------------------------------------------------ #
    # Velocità                                                            #
    # ------------------------------------------------------------------ #
    normgradphi = np.sqrt(dphidxp**2 + dphidyp**2 + dphidzp**2)
    vnormp = np.abs(dphidtp) / normgradphi          # |v| = |∂φ/∂t| / ‖∇φ‖

    v = {"vnormp": vnormp}

    if not speedonlyflag:
        # v⃗ = |v| · (−∇φ / ‖∇φ‖)
        v["vxp"] = vnormp * (-dphidxp / normgradphi)
        v["vyp"] = vnormp * (-dphidyp / normgradphi)
        v["vzp"] = vnormp * (-dphidzp / normgradphi)

    print("v done...")
    return v

