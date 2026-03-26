"""
run_pipeline.py
===============
Script di avvio della pipeline EDF → fase → flusso di fase.

Uso:
    python run_pipeline.py
"""

from main_phaseflow import run_pipeline

if __name__ == "__main__":
    results = run_pipeline(
        "sub-001_ses-pre_task-rest_eeg.edf",
        fmin=8,
        fmax=13,
    )

    vnormp = results["v"]["vnormp"]   # (T, N) velocità scalare
    vxp    = results["v"]["vxp"]      # (T, N) componente x
    loc    = results["loc"]           # (N, 3) coordinate mm

    print(f"\nRisultati disponibili in 'results':")
    print(f"  ch_names : {results['ch_names']}")
    print(f"  vnormp   : shape {vnormp.shape}")
    print(f"  vxp      : shape {vxp.shape}")
    print(f"  loc      : shape {loc.shape}")