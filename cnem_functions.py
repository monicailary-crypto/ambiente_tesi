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

# Imposta a False se lavori con dati 2-D puri (coordi XY già planari)
USE_PCA_PROJECTION = True


# =========================================================================== #
#  UTILITÀ INTERNE                                                             #
# =========================================================================== #

def _prepare_cnem2d_inputs(xyz: np.ndarray, boundary_facets=None):
    """
    Prepara gli input nel formato richiesto da cnem2d.SCNI_CNEM2D.

    cnem2d si aspetta:
        XY_Noeud      : tuple di float, [x0, y0, x1, y1, ...] (2-D flat)
        Nb_Noeud_Front: tuple di int,   numero nodi per ogni contorno
        Ind_Noeud_Front: tuple di int,  indici nodi del contorno (0-based)

    Se xyz è 3-D applica PCA per proiettare sul piano principale,
    poi costruisce il contorno convesso.

    Parametri
    ----------
    xyz : ndarray (N, 2) o (N, 3)
    boundary_facets : ndarray opzionale (M, 2) — segmenti di bordo
        Se None viene calcolato con ConvexHull.

    Ritorna
    -------
    xy_flat  : tuple di float (2*N,) — coordinate 2-D flat
    nb_front : tuple di int   (n_contours,)
    ind_front: tuple di int   (tot_boundary_nodes,)
    pca_axes : ndarray (3,2) o None — per riproiezione in 3-D
    xy2d     : ndarray (N,2) — coordinate 2-D dopo eventuale PCA
    """
    xyz = np.asarray(xyz, dtype=float)

    # --- Proiezione PCA 3D → 2D ---
    if xyz.shape[1] == 3 and USE_PCA_PROJECTION:
        center = xyz.mean(axis=0)
        xc = xyz - center
        _, _, Vt = np.linalg.svd(xc, full_matrices=False)
        pca_axes = Vt[:2].T          # (3, 2): colonne = assi principali
        xy2d = xc @ pca_axes         # (N, 2)
    elif xyz.shape[1] == 2:
        xy2d = xyz
        pca_axes = None
    else:
        raise ValueError("xyz deve avere 2 o 3 colonne.")

    N = xy2d.shape[0]

    # --- Flat layout [x0,y0,x1,y1,...] ---
    xy_flat = tuple(xy2d.ravel(order='C').tolist())

    # --- Contorno di bordo ---
    if boundary_facets is not None:
        # boundary_facets: (M, 2) — segmenti di bordo già estratti
        # (prodotti da _triangles_to_boundary_segments su triangolazione 3D)
        bdy_indices = _order_boundary_segments(boundary_facets)
        nb_front  = (len(bdy_indices),)
        ind_front = tuple(int(i) for i in bdy_indices)
    else:
        # Contorno convesso 2D calcolato direttamente sui punti proiettati.
        # NOTA: ConvexHull in 2D restituisce simplices di shape (M, 2),
        # ovvero segmenti — NON triangoli. hull.vertices contiene già
        # gli indici del contorno ordinati, pronti per cnem2d.
        if N >= 3:
            hull = ConvexHull(xy2d)
            bdy_indices = [int(i) for i in hull.vertices]
        else:
            bdy_indices = list(range(N))
        nb_front  = (len(bdy_indices),)
        ind_front = tuple(bdy_indices)

    return xy_flat, nb_front, ind_front, pca_axes, xy2d


def _order_boundary_segments(segments: np.ndarray):
    """
    Ordina i segmenti di bordo (M, 2) in un contorno chiuso di indici nodo.
    Usato per convertire i triangoli di bordo in una lista ordinata per cnem2d.
    """
    # Costruiamo un grafo di adiacenza
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in segments:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    # Percorso euleriano semplice (ogni nodo ha grado 2 in un contorno chiuso)
    start = int(segments[0, 0])
    path = [start]
    visited_edges = set()
    current = start
    while True:
        neighbours = adj[current]
        moved = False
        for nxt in neighbours:
            edge = (min(current, nxt), max(current, nxt))
            if edge not in visited_edges:
                visited_edges.add(edge)
                path.append(nxt)
                current = nxt
                moved = True
                break
        if not moved or current == start:
            break
    # Rimuovi il nodo di chiusura duplicato
    if path[-1] == path[0]:
        path = path[:-1]
    return path


def _parse_scni_output(scni_result, N: int, pca_axes=None):
    """
    Interpreta l'output di cnem2d.SCNI_CNEM2D e costruisce la matrice B.

    NOTA IMPORTANTE sugli indici:
    Il sorgente scni_cnem2d.cpp aggiunge +1 agli indici prima di restituirli:
        P_Vec_Ind_Noeud_New_Old->push_back(ML.Vec_Ind_Noeud_New_Old[i] + 1)
        P_Vec_Ind_Noeud_Old_New->push_back(ML.Vec_Ind_Noeud_Old_New[i] + 1)
    Quindi new_old e old_new sono 1-based → sottraiamo 1 prima di usarli.

    Analogamente Vec_INV contiene indici nodo 1-based → sottraiamo 1.

    Output di SCNI_CNEM2D (indici 0-7):
        0: Ind_Noeud_New_Old  — permutazione new→old  (1-based)
        1: Ind_Noeud_Old_New  — permutazione old→new  (1-based)
        2: Vol_Cel            — volumi celle di Voronoi
        3: XY_CdM             — centri di massa (flat 2*N)
        4: Nb_Contrib         — numero di contributi per nodo (new ordering)
        5: INV                — indici vicini flat (1-based)
        6: Grad               — gradienti NEM (valori reali, 0-based)
        7: Tri                — triangolazione

    Returns
    -------
    B : ndarray (3*N, N) se pca_axes non è None, altrimenti (2*N, N)
    new_old : ndarray (N,) — mappa indici new→old (0-based)
    old_new : ndarray (N,) — mappa indici old→new (0-based)
    """
    # Indici 1-based → converti a 0-based sottraendo 1
    new_old    = np.array(scni_result[0], dtype=int) - 1   # (N,) 0-based
    old_new    = np.array(scni_result[1], dtype=int) - 1   # (N,) 0-based
    nb_contrib = np.array(scni_result[4], dtype=int)       # (N,) — conteggi, non indici
    inv        = np.array(scni_result[5], dtype=int) - 1   # lista vicini 0-based
    grad_flat  = np.array(scni_result[6], dtype=float)     # gradienti (valori float, non indici)

    # I guard C++ restituiscono 0 per celle degeneri → dopo -1 diventano -1.
    # Clippiamo a [0, N-1]: le righe/colonne corrispondenti avranno contributo zero in B.
    new_old = np.clip(new_old, 0, N - 1)
    old_new = np.clip(old_new, 0, N - 1)
    inv     = np.clip(inv,     0, N - 1)

    # Costruisci B_2d (2*N, N)
    B_2d = np.zeros((2 * N, N))

    pos_inv  = 0
    pos_grad = 0
    for i_new in range(len(nb_contrib)):
        nc    = nb_contrib[i_new]
        i_old = new_old[i_new]      # indice 0-based nel sistema originale
        for _ in range(nc):
            j_new  = inv[pos_inv]   # 0-based
            j_old  = new_old[j_new] if j_new < len(new_old) else 0
            gx     = grad_flat[pos_grad]
            gy     = grad_flat[pos_grad + 1]
            B_2d[0 * N + i_old, j_old] += gx   # ∂/∂x
            B_2d[1 * N + i_old, j_old] += gy   # ∂/∂y
            pos_inv  += 1
            pos_grad += 2

    if pca_axes is None:
        return B_2d, new_old, old_new
    else:
        # Riproiezione 2D → 3D tramite PCA
        B_3d = np.zeros((3 * N, N))
        for d in range(3):
            B_3d[d*N:(d+1)*N, :] = (
                pca_axes[d, 0] * B_2d[0*N:1*N, :]
                + pca_axes[d, 1] * B_2d[1*N:2*N, :]
            )
        return B_3d, new_old, old_new

