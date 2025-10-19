# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules




datas = []
hiddenimports = []

def add_coqui_files(base_path):
    """Helper function to add all COQUI_AI_TTS Python and JSON files recursively."""
    coqui_path = Path(base_path)
    if not coqui_path.exists():
        return
    
    for root, dirs, files in os.walk(coqui_path):
        for file in files:
            if file.endswith(('.py', '.json')):
                src_file = Path(root) / file
                rel_path = src_file.relative_to(coqui_path)
                dest_path = f"COQUI_AI_TTS/{str(rel_path).replace(os.sep, '/')}"
                #print(f"Adding COQUI file: {src_file} -> {os.path.dirname(dest_path)}")
                datas.append((str(src_file), os.path.dirname(dest_path)))

# Include all COQUI_AI_TTS files recursively
add_coqui_files("skyrimnet-xtts/COQUI_AI_TTS")

# Core application data files with better exclusions
datas += collect_data_files("gradio_client", excludes=[
    "*.md", "*.txt", "*.rst", "test*", "*test*", "example*", "*example*"
])
datas += collect_data_files("gradio", excludes=[
    "*.md", "*.txt", "*.rst", "test*", "*test*", "demo*"
])

# Keep these smaller libraries as-is (minimal benefit to optimize)
datas += collect_data_files("groovy")
datas += collect_data_files("safehttpx")

# CRITICAL: Include comprehensive spaCy data files
import os
import spacy

MODEL_SUPPORTED_LANGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja"] # Supported languages from XTTS v2 model config.json
SPACY_REQUIRED_LANGS = ['ar','en','es','hi','ja','zh','ko','vi','th'] 

required_langs = SPACY_REQUIRED_LANGS #MODEL_SUPPORTED_LANGS + SPACY_REQUIRED_LANGS
required_langs = [lang.split("-")[0] for lang in required_langs]  # Normalize to primary language codes
required_langs = list(set(required_langs))  # Remove duplicates
spacy_lang_path = os.path.join(os.path.dirname(spacy.__file__), 'lang')

for lang_code in required_langs:
    lang_dir = os.path.join(spacy_lang_path, lang_code)
    if os.path.exists(lang_dir):
        #print(f"Found language directory: {lang_dir}")
        
        for root, dirs, files in os.walk(lang_dir):
            for file in files:
                if file.endswith('.py'):
                    src_path = os.path.join(root, file)

                    rel_path = os.path.relpath(src_path, spacy_lang_path)
                    dest_path = f"spacy/lang/{rel_path.replace(os.sep, '/')}"
                    datas.append((src_path, os.path.dirname(dest_path)))
                    #print(f"  Added: {src_path} -> {dest_path}")
    else:
        print(f"Warning: Language directory not found: {lang_dir}")
        pass

try:
    core_spacy_data = collect_data_files("spacy", excludes=["*.pyc", "__pycache__"])
    datas.extend(core_spacy_data)
except Exception as e:
    print(f"Warning: Core spaCy data collection failed: {e}")


#print(f"Total spaCy data files added: {len([d for d in datas if 'spacy' in d[1]])}")

# Include setuptools data files (needed for jaraco.text and other components)
datas += collect_data_files("setuptools", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "docs/*", "*/docs/*"
])

datas += collect_data_files("unidic_lite", excludes=[
    "test*", "*test*", "*.md", "*.txt", "*.rst"
])
datas += collect_data_files("jamo", excludes=[
    "test*", "*test*", "*.md", "*.txt", "*.rst"
])

datas += collect_data_files("hangul_romanize", excludes=[
    "test*", "*test*", "*.md", "*.txt", "*.rst"
])

datas += collect_data_files("cutlet", excludes=[
    "test*", "*test*", "*.md", "*.txt", "*.rst"
])


# CRITICAL: Only include essential CuPy components (if available)
try:
    # More selective CuPy inclusion to avoid massive libraries
    datas += collect_data_files("cupy", excludes=[
        "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
        "test*", "*test*", "tests/*", "*/tests/*",
        "example*", "*example*", "examples/*", "*/examples/*",
        "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*",
        # Exclude huge CUDA libraries that are often unnecessary
        "*.lib", "lib/*", "libs/*"
    ])
    
    # Only essential CUDA backends 
    datas += collect_data_files("cupy_backends", excludes=[
        "*.cpp", "*.cu", "*.c", "*.h", "*.cuh", "test*", "*test*",
        "*.lib", "lib/*", "libs/*"
    ])
    
    # Be more selective about CUDA components
    datas += collect_data_files("cupy_backends.cuda", excludes=[
        "*.lib", "lib/*", "libs/*", "test*", "*test*"
    ])
except:
    pass

# CRITICAL: PyTorch with enhanced exclusions to prevent bloat
datas += collect_data_files("torch", excludes=[
    "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*",
    # CRITICAL: Exclude massive library files that cause bloat
    "*.lib", "lib/*.lib", "libs/*.lib",
    # Exclude huge CUDA runtime libraries
    "lib/libtorch_cuda.so*", "lib/libtorch_cpu.a", "lib/dnnl.lib",
    # Multi-GPU solvers (not needed for single-GPU TTS)
    "lib/cusolverMg64_11.dll",   # 179 MB - Multi-GPU solvers
    # NOTE: cuDNN and other CUDA DLL exclusions are handled in post-processing
    # because collect_data_files() doesn't reliably exclude binary dependencies
])

datas += collect_data_files("deepspeed", excludes=[
    "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*",
])

datas += collect_data_files("torchaudio", excludes=[
    "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
])

datas += collect_data_files("spacy", excludes=[
    "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
])

datas += collect_data_files("transformers", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
])

# CRITICAL: Include Triton backend data files (driver.py and other backend modules)
# Exclude AMD backend entirely - we only need NVIDIA CUDA backend
datas += collect_data_files("triton", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    "example*", "*example*", "examples/*", "*/examples/*",
    "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*",
    "backends/amd/*",  # Exclude AMD backend
    "backends/amd"
])

# =============================================================================
# OPTIMIZED HIDDEN IMPORTS (Keep original structure but reduce scope)
# =============================================================================

# Hidden imports for CUDA backends and core modules (keep essential only)
try:
    import cupy
    hiddenimports += collect_submodules("cupy_backends.cuda")
except ImportError:
    pass


# CRITICAL: Force inclusion of ALL spaCy language modules
try:
    # First try to collect all spacy.lang modules
    hiddenimports += collect_submodules('spacy.lang')
    
    # Then force specific language modules that TTS needs
    for lang_code in required_langs: # ['en', 'es', 'ar', 'hi', 'ja', 'zh', 'ko', 'vi', 'th']:
        try:
            hiddenimports += collect_submodules(f'spacy.lang.{lang_code}')
        except:
            print(f"Warning: Could not collect submodules for spacy.lang.{lang_code}")
            
except Exception as e:
    print(f"Warning: spaCy submodule collection failed: {e}")

# CRITICAL: Explicit spaCy language modules needed by TTS tokenizer
hiddenimports += [
    'spacy.lang.ar',
    'spacy.lang.en',
    'spacy.lang.es', 
    'spacy.lang.hi',
    'spacy.lang.ja',
    'spacy.lang.zh',

]

# Essential PyTorch modules only
hiddenimports += collect_submodules('torch.nn.functional')

hiddenimports += collect_submodules('deepspeed')

# Fix torch._dynamo import issues (keep minimal)
hiddenimports += collect_submodules('torch._dynamo.polyfills')

# Fix transformers import issues (reduce scope)
hiddenimports += collect_submodules('transformers.generation.utils')
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

# Language processing modules (only essential)
hiddenimports += [
    'unidic_lite', 
    'fugashi',
    'jamo',
    'jaraco',
    'setuptools',
    # CRITICAL: Core spaCy modules needed for TTS
    'spacy',
    'spacy.tokens',
    'spacy.lang',
    'spacy.pipeline',
    'spacy.pipeline.sentencizer',
]


# Add language-specific modules only if needed
if any(lang.startswith("zh") for lang in MODEL_SUPPORTED_LANGS+SPACY_REQUIRED_LANGS):
    hiddenimports += ['spacy_pkuseg', 'pypinyin']
if "ko" in MODEL_SUPPORTED_LANGS+SPACY_REQUIRED_LANGS:
    hiddenimports += ['g2pkk', 'jamo']
if "ja" in MODEL_SUPPORTED_LANGS+SPACY_REQUIRED_LANGS:
    hiddenimports += ['unidic_lite']

# Fix specific import errors (minimal set)
hiddenimports += [
    'torch._dynamo.polyfills.fx',
    "triton",
    "triton.compiler",
    "triton.tools",
    "triton.language",
    "triton.backends.nvidia",
    "triton.backends.nvidia.driver",
    'deepspeed',
]

# =============================================================================
# EXCLUDE PROBLEMATIC MODULES (from original)
# =============================================================================

excludedimports = [
    'ninja',
    #'torch.utils.cpp_extension',
    # Exclude AMD backend (we only use NVIDIA CUDA)
    'triton.backends.amd',
    # Exclude unsupported language modules for spaCy (based on XTTS v2 supported languages)
    # Supported by TTS: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi
    # CRITICAL: Do NOT exclude languages that TTS actually uses: en, es, ar, hi, ja, zh
    # Exclude all others:
    'spacy.lang.af',   # Afrikaans
    'spacy.lang.am',   # Amharic  
    'spacy.lang.az',   # Azerbaijani
    'spacy.lang.bg',   # Bulgarian
    'spacy.lang.bn',   # Bengali
    'spacy.lang.bs',   # Bosnian
    'spacy.lang.ca',   # Catalan
    'spacy.lang.cy',   # Welsh
    'spacy.lang.da',   # Danish
    'spacy.lang.el',   # Greek
    'spacy.lang.eo',   # Esperanto
    'spacy.lang.et',   # Estonian
    'spacy.lang.eu',   # Basque
    'spacy.lang.fa',   # Persian
    'spacy.lang.fi',   # Finnish
    'spacy.lang.ga',   # Irish
    'spacy.lang.gl',   # Galician
    'spacy.lang.gu',   # Gujarati
    'spacy.lang.he',   # Hebrew
    'spacy.lang.hr',   # Croatian
    'spacy.lang.hy',   # Armenian
    'spacy.lang.id',   # Indonesian
    'spacy.lang.is',   # Icelandic
    'spacy.lang.kn',   # Kannada
    'spacy.lang.ky',   # Kyrgyz
    'spacy.lang.lb',   # Luxembourgish
    'spacy.lang.lij',  # Ligurian
    'spacy.lang.lt',   # Lithuanian
    'spacy.lang.lv',   # Latvian
    'spacy.lang.mk',   # Macedonian
    'spacy.lang.ml',   # Malayalam
    'spacy.lang.mr',   # Marathi
    'spacy.lang.nb',   # Norwegian Bokmal
    'spacy.lang.ne',   # Nepali
    'spacy.lang.ro',   # Romanian
    'spacy.lang.sa',   # Sanskrit
    'spacy.lang.si',   # Sinhala
    'spacy.lang.sk',   # Slovak
    'spacy.lang.sl',   # Slovenian
    'spacy.lang.sq',   # Albanian
    'spacy.lang.sr',   # Serbian
    'spacy.lang.sv',   # Swedish
    'spacy.lang.ta',   # Tamil
    'spacy.lang.te',   # Telugu
    #'spacy.lang.th',   # Thai
    'spacy.lang.ti',   # Tigrinya
    'spacy.lang.tl',   # Tagalog
    'spacy.lang.tn',   # Tswana
    'spacy.lang.tt',   # Tatar
    'spacy.lang.uk',   # Ukrainian
    'spacy.lang.ur',   # Urdu
    #'spacy.lang.vi',   # Vietnamese
    'spacy.lang.yo',   # Yoruba
]

# =============================================================================
# ANALYSIS (Keep original configuration mostly)
# =============================================================================

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
        "pyinstaller-hooks/rthook_setup_cuda_path.py",
    ],
    excludes=excludedimports,
    noarchive=False,
    optimize=1,  # Conservative optimization
    module_collection_mode={ 
        'skyrimnet-xtts.COQUI_AI_TTS.configs': 'py+pyz',
        'skyrimnet-xtts.COQUI_AI_TTS.vocoder.models': 'py+pyz',
        'skyrimnet-xtts.COQUI_AI_TTS.vocoder.layers': 'py+pyz',
        'gradio': 'py+pyz',
        'torch': 'py+pyz',
        'deepspeed': 'py+pyz',
    },
    cipher=None,
    upx=True,
)

# =============================================================================
# NVIDIA CUDA DLL EXCLUSION - Remove system-provided CUDA libraries
# =============================================================================
# These DLLs will be loaded from the system CUDA installation
# CUDA_PATH environment variable points to: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
# DLLs are available in: %CUDA_PATH%\bin\

cuda_dlls_to_exclude = [
    # ULTRA-CONSERVATIVE APPROACH: Only exclude the absolutely safe CUDA runtime DLL
    # Testing shows shm.dll loading issues - reducing exclusions to minimal set
    
    # CUDA Runtime (available in system CUDA installation)
    'cudart64_12.dll',                          # 0.6 MB - CUDA Runtime - SAFE TO EXCLUDE
    
    # Temporarily removing other exclusions to debug shm.dll loading issue
    # Will re-add after confirming application starts correctly
    
    # NOTE: Keep these PyTorch-required DLLs that were previously excluded:
    'cublas64_12.dll', # (97.8 MB) - Required by torch_cuda.dll
    'cublaslt64_12.dll', # (638 MB) - Required by torch_cuda.dll
    'cufft64_11.dll', # (274 MB) - Required by torch_cuda.dll
    'cufftw64_11.dll', # (0.2 MB) - FFTW Interface
    'curand64_10.dll', # (75.5 MB) - Random Number Generation
    'cusolver64_11.dll', # (270 MB) - Required by torch_cuda.dll
    'cusparse64_12.dll', # (455.4 MB) - Required by torch_cuda.dll
    'nvrtc64_120_0.dll', # (85.7 MB) - Runtime Compilation
    # - cudnn64_9.dll (0.3 MB) - Required by torch_cuda.dll
    # - cudnn_cnn64_9.dll (4.4 MB) - Core CNN operations
    # - cudnn_ops64_9.dll (120.6 MB) - Core operations
    # - cudnn_engines_runtime_compiled64_9.dll (19.3 MB) - Runtime engines
    # - cudnn_graph64_9.dll (2.3 MB) - Graph operations
]

# Additional post-processing to remove bloat and CUDA system libraries
a.datas = [x for x in a.datas if not any([
    # Remove .lib files (redundant with collect_data_files exclusions, but some may slip through)
    x[0].lower().endswith('.lib') and 'dnnl' in x[0].lower(),
    x[0].lower().endswith('.lib') and any(huge in x[0].lower() for huge in ['cublas', 'cudnn', 'cufft', 'cusolver']),
    
    # CRITICAL: Exclude NVIDIA CUDA system DLLs - users will have these installed
    any(cuda_dll.lower() in x[0].lower() for cuda_dll in cuda_dlls_to_exclude),
])]

# CRITICAL: Remove NVIDIA CUDA DLLs from binaries as well (they get pulled in as binary dependencies)
a.binaries = [x for x in a.binaries if not any([
    # Exclude all NVIDIA CUDA system DLLs - users will have these installed via CUDA toolkit
    any(cuda_dll.lower() in x[0].lower() for cuda_dll in cuda_dlls_to_exclude),
])]

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# =============================================================================
# EXECUTABLE CONFIGURATION (Keep original structure)
# =============================================================================

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='skyrimnet-xtts',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Disable strip to avoid Windows issues
    upx=False,    # Disable UPX for now to avoid issues
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
    strip=False,   # Disable strip
    upx=False,     # Disable UPX compression to avoid massive files
    upx_exclude=[],
    name='skyrimnet-xtts',
    optimize=2,
)