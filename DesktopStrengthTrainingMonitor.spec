# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_app_files.py'],
    pathex=[],
    binaries=[],
    datas=[('icons', 'icons'), ('barbell_yolo26n.pt', '.'), ('model_spinal_9pts.pt', '.'), ('yolo26l-pose.pt', '.'), ('artifacts/all_sequence_models_benchmark', 'artifacts/all_sequence_models_benchmark')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch.testing', 'torch.testing._internal', 'matplotlib.tests', 'scipy', 'sphinx', 'IPython', 'jupyter', 'tensorflow', 'keras', 'onnxruntime', 'torchaudio', 'botocore', 'openpyxl', 'sqlalchemy', 'pyarrow', 'h5py'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DesktopStrengthTrainingMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons/app_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DesktopStrengthTrainingMonitor',
)
