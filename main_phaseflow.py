"""
main_phaseflow.py
=================
Pipeline completa:
  1. Caricamento file EDF
  2. Preprocessing (filtro banda alpha, rimozione artefatti)
  3. Fase istantanea (Hilbert)
  4. Flusso di fase 3-D (CNEM)

Requisiti:
    pip install numpy scipy
    + cnem2d.pyd / cnem2d.so nel PYTHONPATH

Struttura file prevista nella stessa cartella:
    brain_functions.py       (phases_nodes, grad_B_cnem, grad_cnem, phaseflow_cnem)
    cnem_functions.py        (_prepare_cnem2d_inputs, _parse_scni_output, ...)
    main_phaseflow.py        ← questo file

Uso:
    python main_phaseflow.py
    python main_phaseflow.py --edf percorso/al/file.edf
    python main_phaseflow.py --edf file.edf --fmin 8 --fmax 13 --tmin 0 --tmax 60
"""

import argparse
import struct
import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

from cnem_functions import _prepare_cnem2d_inputs
from cnem_functions import _order_boundary_segments
from cnem_functions import _parse_scni_output

from brain_functions import phases_nodes
from brain_functions import grad_B_cnem
from brain_functions import _triangles_to_boundary_segments
from brain_functions import grad_cnem
from brain_functions import phaseflow_cnem


# =========================================================================== #
#  PARAMETRI DEFAULT                                                           #
# =========================================================================== #

DEFAULT_EDF   = "sub-001_ses-pre_task-rest_eeg.edf"
DEFAULT_FMIN  = 8.0    # Hz — limite inferiore banda (default: alpha)
DEFAULT_FMAX  = 13.0   # Hz — limite superiore banda
DEFAULT_TMIN  = None   # s  — inizio segmento (None = tutto)
DEFAULT_TMAX  = None   # s  — fine segmento   (None = tutto)
# Elettrodi da escludere (nasofaringei, EOG, EMG, riferimento, ...)
EXCLUDE_CHANNELS = ["Pg1", "Pg2", "A1", "A2", "M1", "M2"]

# Raggio testa (mm) per il montaggio 10-20 hardcoded
HEAD_RADIUS_MM = 85.0

# =========================================================================== #
#  COORDINATE 10-20 STANDARD (sferiche → cartesiane)                          #
# =========================================================================== #
# Formato: label → (theta_deg, phi_deg)
# x = R·sin(φ)·cos(θ),  y = R·sin(φ)·sin(θ),  z = R·cos(φ)
_COORDS_10_20_SPH = {
    "Fp1": (-18, 72),  "Fp2": (18, 72),
    "F7":  (-54, 72),  "F3":  (-36, 54),  "Fz":  (0, 54),   "F4":  (36, 54),   "F8":  (54, 72),
    "T3":  (-90, 90),  "C3":  (-54, 90),  "Cz":  (0, 90),   "C4":  (54, 90),   "T4":  (90, 90),
    "T5":  (-54, 126), "P3":  (-36, 126), "Pz":  (0, 126),  "P4":  (36, 126),  "T6":  (54, 126),
    "O1":  (-18, 144), "O2":  (18, 144),
    # Alias maiuscoli (come appaiono nell'EDF)
    "FZ":  (0, 54),  "CZ":  (0, 90),  "PZ":  (0, 126),
}


def _sph_to_cart(theta_deg: float, phi_deg: float, r: float = HEAD_RADIUS_MM):
    """Coordinate sferiche (gradi) → cartesiane (mm)."""
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    x = r * np.sin(ph) * np.cos(th)
    y = r * np.sin(ph) * np.sin(th)
    z = r * np.cos(ph)
    return np.array([x, y, z])


def get_standard_locations(channel_names: list) -> tuple[np.ndarray, list]:
    """
    Restituisce le coordinate 3-D (mm) per i canali con posizione nota.

    Ritorna
    -------
    loc      : ndarray (N_valid, 3)
    ch_valid : list di str — nomi canali con coordinata disponibile
    """
    ch_valid = []
    coords   = []
    for ch in channel_names:
        key = ch.strip()
        if key in _COORDS_10_20_SPH:
            th, ph = _COORDS_10_20_SPH[key]
            coords.append(_sph_to_cart(th, ph))
            ch_valid.append(ch)
        else:
            print(f"  [WARN] Nessuna coordinata standard per '{ch}' → escluso")
    return np.array(coords), ch_valid


# =========================================================================== #
#  LETTURA EDF (senza MNE)                                                     #
# =========================================================================== #

def read_edf(filepath: str) -> dict:
    """
    Legge un file EDF e restituisce un dizionario con:
        'labels'   : list of str
        'sfreq'    : float (Hz)
        'n_records': int
        'duration' : float (s per record)
        'data'     : ndarray (n_channels, n_samples) in uV
        'phys_min' : list of float
        'phys_max' : list of float
        'dig_min'  : list of float
        'dig_max'  : list of float

    Supporta EDF standard (non EDF+).
    """
    with open(filepath, "rb") as f:
        # ---- Header globale (256 byte) ----
        hdr = f.read(256)
        n_bytes_hdr  = int(hdr[184:192].decode("ascii", errors="replace").strip())
        n_records    = int(hdr[236:244].decode("ascii", errors="replace").strip())
        duration_rec = float(hdr[244:252].decode("ascii", errors="replace").strip())
        n_signals    = int(hdr[252:256].decode("ascii", errors="replace").strip())

        # ---- Signal header (n_signals * 256 byte) ----
        ns = n_signals
        sh = f.read(ns * 256)

        def _field(offset, width, cast=str):
            items = []
            for i in range(ns):
                raw = sh[offset + i * width: offset + (i + 1) * width]
                val = raw.decode("ascii", errors="replace").strip()
                items.append(cast(val) if cast is not str else val)
            return items

        labels    = _field(0,         16)
        phys_min  = _field(ns * 104,   8, float)
        phys_max  = _field(ns * 112,   8, float)
        dig_min   = _field(ns * 120,   8, float)
        dig_max   = _field(ns * 128,   8, float)
        n_samp    = _field(ns * 216,   8, int)

        sfreq = n_samp[0] / duration_rec

        # Gain e offset per la conversione digitale → fisico
        gain   = [(phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i])
                  for i in range(ns)]
        offset = [phys_min[i] - gain[i] * dig_min[i] for i in range(ns)]

        # ---- Dati ----
        # Ogni record contiene n_samp[i] campioni int16 per il canale i
        data_per_ch = [[] for _ in range(ns)]

        for _ in range(n_records):
            for i in range(ns):
                raw_bytes = f.read(n_samp[i] * 2)
                samples   = struct.unpack(f"<{n_samp[i]}h", raw_bytes)
                data_per_ch[i].extend(samples)

    # Conversione digitale → fisico (uV)
    data = np.array([
        np.array(data_per_ch[i], dtype=np.float64) * gain[i] + offset[i]
        for i in range(ns)
    ])  # shape: (n_channels, n_samples)

    return {
        "labels":    labels,
        "sfreq":     sfreq,
        "n_records": n_records,
        "duration":  duration_rec * n_records,
        "data":      data,       # (n_ch, n_samples) uV
        "phys_min":  phys_min,
        "phys_max":  phys_max,
    }


# =========================================================================== #
#  PREPROCESSING                                                               #
# =========================================================================== #

def bandpass_filter(data: np.ndarray, sfreq: float,
                    fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    """
    Filtro Butterworth passa-banda applicato su ogni canale.

    Parametri
    ----------
    data  : ndarray (n_ch, n_samples)
    sfreq : float
    fmin, fmax : float — frequenze di taglio in Hz
    order : int

    Ritorna
    -------
    filtered : ndarray (n_ch, n_samples)
    """
    nyq = sfreq / 2.0
    low  = fmin / nyq
    high = fmax / nyq
    low  = np.clip(low,  1e-4, 0.9999)
    high = np.clip(high, 1e-4, 0.9999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=1)


def select_segment(data: np.ndarray, sfreq: float,
                   tmin: float = None, tmax: float = None) -> np.ndarray:
    """
    Ritaglia il segnale tra tmin e tmax (in secondi).
    Se tmin/tmax sono None usa l'intera registrazione.
    """
    n_samples = data.shape[1]
    i_start = int(tmin * sfreq) if tmin is not None else 0
    i_end   = int(tmax * sfreq) if tmax is not None else n_samples
    i_start = max(0, i_start)
    i_end   = min(n_samples, i_end)
    return data[:, i_start:i_end]


# =========================================================================== #
#  MAIN PIPELINE                                                               #
# =========================================================================== #

def run_pipeline(edf_path:  str,
                 fmin:      float = DEFAULT_FMIN,
                 fmax:      float = DEFAULT_FMAX,
                 tmin:      float = None,
                 tmax:      float = None,
                 speed_only: bool = False) -> dict:
    """
    Esegue la pipeline completa e restituisce i risultati.

    Ritorna
    -------
    results : dict con chiavi
        'ch_names'  – list of str: nomi canali usati
        'loc'       – ndarray (N, 3): coordinate mm
        'sfreq'     – float
        'yphasep'   – ndarray (T, N): fasi istantanee (rad)
        'v'         – dict: output di phaseflow_cnem
                        'vnormp', ['vxp', 'vyp', 'vzp']
        'dt'        – float: passo temporale (s)
        'n_samples' – int: campioni analizzati
    """
    # ------------------------------------------------------------------ #
    # STEP 1: Caricamento EDF                                             #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"STEP 1 — Caricamento EDF")
    print(f"{'='*60}")
    print(f"  File   : {edf_path}")

    edf = read_edf(edf_path)

    print(f"  Canali : {len(edf['labels'])}")
    print(f"  sfreq  : {edf['sfreq']} Hz")
    print(f"  Durata : {edf['duration']:.1f} s ({edf['duration']/60:.1f} min)")
    print(f"  Shape  : {edf['data'].shape}")

    # ------------------------------------------------------------------ #
    # STEP 2: Selezione canali EEG validi                                 #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"STEP 2 — Selezione canali e coordinate")
    print(f"{'='*60}")

    # Rimuovi canali esclusi (nasofaringei, ECG, ecc.)
    all_labels = edf["labels"]
    keep_mask  = [lbl.strip() not in EXCLUDE_CHANNELS for lbl in all_labels]
    labels_kept = [lbl for lbl, k in zip(all_labels, keep_mask) if k]
    data_kept   = edf["data"][[i for i, k in enumerate(keep_mask) if k], :]

    print(f"  Canali dopo esclusione: {len(labels_kept)}  {labels_kept}")

    # Recupera coordinate 3-D standard 10-20
    loc, ch_valid = get_standard_locations(labels_kept)

    # Aggiungi questo nel main_phaseflow.py prima di far partire tutto

    if len(ch_valid) == 0:
        raise RuntimeError("Nessun canale con coordinate 3-D disponibile.")

    # Seleziona solo i canali con coordinate
    valid_idx  = [labels_kept.index(ch) for ch in ch_valid]
    data_valid = data_kept[valid_idx, :]

    print(f"  Canali con coordinate 3-D: {len(ch_valid)}")
    print(f"  Coordinate loc shape: {loc.shape}")

    # ------------------------------------------------------------------ #
    # STEP 3: Selezione segmento temporale                                #
    # ------------------------------------------------------------------ #
    if tmin is not None or tmax is not None:
        print(f"\n{'='*60}")
        print(f"STEP 3 — Ritaglio segmento [{tmin or 0:.1f}s, {tmax or edf['duration']:.1f}s]")
        print(f"{'='*60}")
        data_valid = select_segment(data_valid, edf["sfreq"], tmin, tmax)
        print(f"  Shape dopo ritaglio: {data_valid.shape}")

    # ------------------------------------------------------------------ #
    # STEP 4: Filtro passa-banda                                          #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"STEP 4 — Filtro passa-banda [{fmin:.1f}–{fmax:.1f} Hz]")
    print(f"{'='*60}")
    data_filtered = bandpass_filter(data_valid, edf["sfreq"], fmin, fmax)
    print(f"  Shape filtrata: {data_filtered.shape}")

    # ------------------------------------------------------------------ #
    # STEP 5: Fase istantanea                                             #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"STEP 5 — Fase istantanea (Hilbert + unwrap)")
    print(f"{'='*60}")

    # phases_nodes si aspetta (T, N): trasponiamo
    yp = data_filtered.T   # (T, N)

    from brain_functions import phases_nodes
    yphasep = phases_nodes(yp)   # (T, N)
    print(f"  yphasep shape : {yphasep.shape}")
    print(f"  range fase    : [{yphasep.min():.2f}, {yphasep.max():.2f}] rad")

    # ------------------------------------------------------------------ #
    # STEP 6: Flusso di fase                                              #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"STEP 6 — Flusso di fase (CNEM)")
    print(f"{'='*60}")

    dt = 1.0 / edf["sfreq"]

    from brain_functions import phaseflow_cnem
    v = phaseflow_cnem(yphasep, loc, dt, speedonlyflag=speed_only)

    print(f"\n  vnormp shape  : {v['vnormp'].shape}")
    vnorm_finite = v['vnormp'][np.isfinite(v['vnormp'])]
    print(f"  velocità mediana : {np.median(vnorm_finite):.4f}  (unità coerenti con loc in mm)")
    print(f"  velocità media   : {np.mean(vnorm_finite):.4f}")
    print(f"  velocità 95°pct  : {np.percentile(vnorm_finite, 95):.4f}")

    if not speed_only:
        for comp in ("vxp", "vyp", "vzp"):
            arr = v[comp][np.isfinite(v[comp])]
            print(f"  {comp} mean={arr.mean():.4f}  std={arr.std():.4f}")

    return {
        "ch_names":  ch_valid,
        "loc":       loc,
        "sfreq":     edf["sfreq"],
        "yphasep":   yphasep,
        "v":         v,
        "dt":        dt,
        "n_samples": data_valid.shape[1],
    }


# =========================================================================== #
#  ENTRY POINT                                                                 #
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline EDF → fase istantanea → flusso di fase (CNEM)"
    )
    parser.add_argument(
        "--edf",   default=DEFAULT_EDF,
        help=f"Percorso file EDF (default: {DEFAULT_EDF})"
    )
    parser.add_argument(
        "--fmin",  type=float, default=DEFAULT_FMIN,
        help=f"Frequenza minima filtro passa-banda in Hz (default: {DEFAULT_FMIN})"
    )
    parser.add_argument(
        "--fmax",  type=float, default=DEFAULT_FMAX,
        help=f"Frequenza massima filtro passa-banda in Hz (default: {DEFAULT_FMAX})"
    )
    parser.add_argument(
        "--tmin",  type=float, default=None,
        help="Inizio segmento in secondi (default: inizio registrazione)"
    )
    parser.add_argument(
        "--tmax",  type=float, default=None,
        help="Fine segmento in secondi (default: fine registrazione)"
    )
    parser.add_argument(
        "--speed-only", action="store_true",
        help="Calcola solo la velocità scalare (più rapido, senza vxp/vyp/vzp)"
    )

    args = parser.parse_args()

    results = run_pipeline(
        edf_path   = args.edf,
        fmin       = args.fmin,
        fmax       = args.fmax,
        tmin       = args.tmin,
        tmax       = args.tmax,
        speed_only = args.speed_only,
    )

    print(f"\n{'='*60}")
    print("Pipeline completata.")
    print(f"  Canali analizzati : {len(results['ch_names'])}")
    print(f"  Campioni          : {results['n_samples']}")
    print(f"  Output in results['v'] — chiavi: {list(results['v'].keys())}")
    print(f"{'='*60}\n")
