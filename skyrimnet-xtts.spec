# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = []
hiddenimports = []

# Core application data files
datas += collect_data_files("gradio_client")
datas += collect_data_files("gradio")
datas += collect_data_files("groovy")
datas += collect_data_files("safehttpx")

# TTS library data files (excluding C++ source files)
try:
    datas += collect_data_files("TTS", excludes=["*.cpp", "*.cu", "*.c", "*.h", "*.cuh"])
    datas += collect_data_files("TTS.vocoder", excludes=["*.cpp", "*.cu", "*.c", "*.h", "*.cuh"])
    datas += collect_data_files("TTS.vocoder.configs")
except:
    pass

# Language processing libraries
datas += collect_data_files("gruut")
datas += collect_data_files("unidic_lite")
datas += collect_data_files("jamo")



# CuPy libraries (if available)
try:
    datas += collect_data_files("cupy", excludes=["*.cpp", "*.cu", "*.c", "*.h", "*.cuh"])
    datas += collect_data_files("cupy_backends")
    datas += collect_data_files("cupy_backends.cuda")
except:
    pass

# PyTorch libraries (excluding C++ source files)
datas += collect_data_files("torch", excludes=["*.cpp", "*.cu", "*.c", "*.h", "*.cuh"])
datas += collect_data_files("torchaudio", excludes=["*.cpp", "*.cu", "*.c", "*.h", "*.cuh"])
datas += collect_data_files("transformers")


# Hidden imports for CUDA backends and core modules
hiddenimports += collect_submodules("cupy_backends.cuda")
hiddenimports += collect_submodules('TTS')
hiddenimports += collect_submodules('torch.nn.functional')

# Fix torch._dynamo import issues
hiddenimports += collect_submodules('torch._dynamo')
hiddenimports += collect_submodules('torch._dynamo.polyfills')

# Fix transformers import issues
hiddenimports += collect_submodules('transformers.generation')
hiddenimports += collect_submodules('transformers.utils')



# Application modules
hiddenimports += [
    "shared_config",
    "shared_models", 
    "shared_args",
    "utils",
    "skyrimnet_api",
    "skyrimnet_xtts"
]

# Language processing modules
hiddenimports += [
    'gruut',
    'unidic_lite', 
    'jamo',
    'jieba',
    'pypinyin',
    'anyascii',
    'inflect',
    'pysbd',

]

# Fix specific import errors
hiddenimports += [
    'torch._dynamo.polyfills.fx',
    'transformers.generation.utils',
    'transformers.generation.utils.GenerationMixin',
]

# Exclude problematic C++ compilation modules
excludedimports = [
    'ninja',
    'torch.utils.cpp_extension',
]

a = Analysis(
    ["skyrimnet-xtts\\__main__.py"],
    pathex=["skyrimnet-xtts"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
                "pyinstaller-hooks/rthook_disable_typeguard.py",
    ],
    excludes=excludedimports,
    noarchive=False,
    optimize=1,
    module_collection_mode={ 
        'gradio': 'py',
        'TTS.vocoder.configs': 'py+pyz',
        'TTS.vocoder.models': 'py+pyz',
        'TTS.vocoder.layers': 'py+pyz',
        'torch': 'py+pyz',
    },
    upx=True,
)

pyz = PYZ(a.pure)

#exe = EXE(
#    pyz,
#    a.scripts,
#    a.binaries,
#    a.datas,
#    [],
#    name="skyrimnet-xtts",
#    debug=False,
#    bootloader_ignore_signals=False,
#    strip=False,
#    upx=True,
#    upx_exclude=[],
#    runtime_tmpdir=None,
#    console=True,
#    disable_windowed_traceback=False,
#    argv_emulation=False,
#    target_arch=None,
#    codesign_identity=None,
#    entitlements_file=None,
#)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='skyrimnet-xtts',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    optimize=2,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='skyrimnet-xtts',
    optimize=2,
)

