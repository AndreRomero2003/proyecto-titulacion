# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['medical_processor_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Incluir librerías necesarias para DICOM comprimido
        ('C:\\Python313\\Lib\\site-packages\\pylibjpeg', 'pylibjpeg'),
        ('C:\\Python313\\Lib\\site-packages\\openjpeg', 'openjpeg'),
        # Ajusta la ruta según tu entorno (ver más abajo)
    ],
    hiddenimports=[
        'pylibjpeg',
        'openjpeg',
        'pydicom.encoders',
        'pydicom.pixel_data_handlers',
        'cv2.ximgproc'  # para filtro guiado
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MedicalImageProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # <-- ¡Importante! Sin consola negra
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)