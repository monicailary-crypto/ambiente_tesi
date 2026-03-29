"""
save_for_matlab.py
==================
Salva i risultati della pipeline Python in formato .mat
per validazione in MATLAB.

Uso:
    python save_for_matlab.py

Output: phaseflow_results.mat
"""

import numpy as np
import scipy.io as sio
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from main_phaseflow import run_pipeline

print("Eseguo pipeline Python...")
results = run_pipeline(
    edf_path   = 'sub-001_ses-pre_task-rest_eeg.edf',
    fmin       = 8.0,
    fmax       = 13.0,
    speed_only = False,
)

v      = results['v']
sfreq  = results.get('sfreq', 200.0)
loc    = results.get('loc')        # (19, 3) coordinate elettrodi
yphasep = results.get('yphasep')   # (T, 19) fasi unwrapped

# Converti in formato MATLAB (colonne = variabili, righe = tempo)
# MATLAB usa convenzione (N_canali x T) o (T x N_canali) — usiamo (T x N)
mat_dict = {
    # Segnale e fasi
    'yphasep' : yphasep,          # (T, N) — fasi unwrapped
    'loc'     : loc,              # (N, 3) — coordinate 3D mm
    'dt'      : 1.0 / sfreq,     # float  — passo temporale
    'sfreq'   : sfreq,            # float
    # Risultati phaseflow
    'vnormp'  : v['vnormp'],      # (T, N) — velocità scalare
    'vxp'     : v['vxp'],         # (T, N) — componente x
    'vyp'     : v['vyp'],         # (T, N) — componente y
    'vzp'     : v['vzp'],         # (T, N) — componente z
    # Info canali
    'channels': np.array(['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
                           'F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'],
                          dtype=object),
}

# Rimuovi None se loc o yphasep non sono stati salvati in results
mat_dict = {k: v for k, v in mat_dict.items() if v is not None}

out_path = 'phaseflow_results.mat'
sio.savemat(out_path, mat_dict)
print(f"\nSalvato: {out_path}")
print(f"Variabili salvate: {list(mat_dict.keys())}")
print(f"  yphasep : {yphasep.shape if yphasep is not None else 'N/A'}")
print(f"  loc     : {loc.shape if loc is not None else 'N/A'}")
print(f"  vnormp  : {v['vnormp'].shape}")
print(f"  vxp/vyp/vzp: {v['vxp'].shape}")
print("\nIn MATLAB:")
print("  load('phaseflow_results.mat')")
print("  yphase_matlab = phases_nodes(yphasep);")
print("  v_matlab = phaseflow_cnem(yphase_matlab, loc, dt);")
print("  % confronta v_matlab.vnormp con vnormp")
