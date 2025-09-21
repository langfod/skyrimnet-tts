# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
import warnings
warnings.filterwarnings("ignore", module='setuptools.*', append=True)
warnings.filterwarnings("ignore", module='numbpysbd.*', append=True)
warnings.filterwarnings("ignore", module='jieba.*', append=True)
warnings.filterwarnings("ignore", module='jamo.*', append=True)
warnings.filterwarnings("ignore", module='g2pkk.*', append=True)


datas = []
hiddenimports = []

# =============================================================================
# IMPROVED DATA COLLECTION (Based on original with better exclusions)
# =============================================================================

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
datas += collect_data_files("safehttpx")

# CRITICAL: Include comprehensive spaCy data files (needed for TTS tokenizer)
import os
import spacy

MODEL_SUPPORTED_LANGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja"] # Supported languages from XTTS v2 model config.json
SPACY_REQUIRED_LANGS = ['ko', 'vi', 'th'] 

required_langs = MODEL_SUPPORTED_LANGS + SPACY_REQUIRED_LANGS
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

# TTS library with enhanced exclusions
try:
    datas += collect_data_files("TTS", excludes=[
        "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
        "test*", "*test*", "tests/*", "*/tests/*",
        "example*", "*example*", "examples/*", "*/examples/*", 
        "*.md", "*.txt", "*.rst", "docs/*", "*/docs/*"
    ])
    datas += collect_data_files("TTS.vocoder", excludes=[
        "*.cpp", "*.cu", "*.c", "*.h", "*.cuh",
        "test*", "*test*", "example*", "*example*"
    ])
    datas += collect_data_files("TTS.vocoder.configs")
except:
    pass

# Language processing libraries with exclusions
datas += collect_data_files("gruut", excludes=[
    "test*", "*test*", "example*", "*example*", "*.md", "*.txt", "*.rst"
])
datas += collect_data_files("unidic_lite", excludes=[
    "test*", "*test*", "*.md", "*.txt", "*.rst"
])
datas += collect_data_files("jamo", excludes=[
    "test*", "*test*", "*.md", "*.txt", "*.rst"
])

# CRITICAL: Include librosa data files (needed for registry.txt and other example data)
datas += collect_data_files("librosa", excludes=[
    "test*", "*test*", "tests/*", "*/tests/*",
    # Removed example exclusions to preserve example_data/registry.txt
    "*.md", "*.rst", "docs/*", "*/docs/*"
    # NOTE: Keep .txt files like registry.txt which are needed at runtime
])

# CRITICAL: Explicitly add librosa example_data files to ensure they're included
import os
import librosa
librosa_path = os.path.dirname(librosa.__file__)
librosa_example_data = os.path.join(librosa_path, 'util', 'example_data')
if os.path.exists(librosa_example_data):
    for file in os.listdir(librosa_example_data):
        if file.endswith(('.txt', '.json', '.py')):
            datas.append((
                os.path.join(librosa_example_data, file),
                'librosa/util/example_data'
            ))

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

# =============================================================================
# OPTIMIZED HIDDEN IMPORTS (Keep original structure but reduce scope)
# =============================================================================

# Hidden imports for CUDA backends and core modules (keep essential only)
try:
    import cupy
    hiddenimports += collect_submodules("cupy_backends.cuda")
except ImportError:
    pass

# TTS modules (reduce to essential only)
hiddenimports += collect_submodules('TTS.tts.configs')
hiddenimports += collect_submodules('TTS.tts.models')
hiddenimports += collect_submodules('TTS.utils')

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
    'spacy.lang.en',
    'spacy.lang.es', 
    'spacy.lang.ar',
    'spacy.lang.hi',
    'spacy.lang.ja',
    'spacy.lang.ko',
    'spacy.lang.vi',
    'spacy.lang.th',
    'spacy.lang.zh',

]

# Essential PyTorch modules only
hiddenimports += collect_submodules('torch.nn.functional')

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
    'gruut',
    'unidic_lite', 
    'jamo',
    'anyascii',
    'inflect',
    'pysbd',
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
    hiddenimports += ['jieba', 'pypinyin']
if "ko" in MODEL_SUPPORTED_LANGS+SPACY_REQUIRED_LANGS:
    hiddenimports += ['g2pkk', 'jamo']
if "ja" in MODEL_SUPPORTED_LANGS+SPACY_REQUIRED_LANGS:
    hiddenimports += ['unidic_lite']

# Fix specific import errors (minimal set)
hiddenimports += [
    'torch._dynamo.polyfills.fx',
    'transformers.generation.utils.GenerationMixin',
]

# =============================================================================
# EXCLUDE PROBLEMATIC MODULES (from original)
# =============================================================================

excludedimports = [
    'ninja',
    'torch.utils.cpp_extension',
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
    # NOTE: spacy.lang.zh is NOT excluded because TTS needs it for Chinese support; th and vi are by spacy lang internals
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
    ],
    excludes=excludedimports,
    noarchive=False,
    optimize=1,  # Conservative optimization
    module_collection_mode={ 
        'gradio': 'py+pyz',
        'TTS.vocoder.configs': 'py+pyz',
        'TTS.vocoder.models': 'py+pyz',
        'TTS.vocoder.layers': 'py+pyz',
        'torch': 'py+pyz',
    },
    cipher=None,
    upx=True,
)

# Additional post-processing to remove bloat
a.datas = [x for x in a.datas if not any([
    # Remove .lib files (redundant with collect_data_files exclusions, but some may slip through)
    x[0].lower().endswith('.lib') and 'dnnl' in x[0].lower(),
    x[0].lower().endswith('.lib') and any(huge in x[0].lower() for huge in ['cublas', 'cudnn', 'cufft', 'cusolver']),
    # CRITICAL: Exclude cuDNN training/optimization DLLs (collect_data_files doesn't handle these reliably)
    # Be conservative - only exclude the largest clearly training-specific libraries
    #'cudnn_engines_precompiled64_9.dll' in x[0].lower(),    # 490 MB
    'cudnn_adv64_9.dll' in x[0].lower(),                    # 269 MB  
    #'cudnn_ops64_9.dll' in x[0].lower(),                    # 121 MB
    #'cudnn_heuristic64_9.dll' in x[0].lower(),              # 54 MB
    # Exclude NVRTC alternative DLL (keep main nvrtc64_120_0.dll)
    'nvrtc64_120_0.alt.dll' in x[0].lower(),
    # OPTIONAL: Uncomment these to exclude additional large libraries (test carefully):
    'cublaslt64_12.dll' in x[0].lower(),    # 638 MB - may be needed for torch.compile
    'cufft64_11.dll' in x[0].lower(),       # 274 MB - may be needed for audio processing  
    'cusolver64_11.dll' in x[0].lower(),    # 270 MB - linear algebra solvers
    'cusolverMg64_11.dll' in x[0].lower(),  # 179 MB - multi-GPU solvers
])]

# CRITICAL: Remove cuDNN DLLs from binaries as well (they get pulled in as binary dependencies)
a.binaries = [x for x in a.binaries if not any([
    # Exclude ONLY the largest cuDNN training/optimization DLLs (be conservative)
    #'cudnn_engines_precompiled64_9.dll' in x[0].lower(),    # 490 MB - precompiled engines
    'cudnn_adv64_9.dll' in x[0].lower(),                    # 269 MB - advanced features  
    #'cudnn_ops64_9.dll' in x[0].lower(),                    # 121 MB - operations library
    #'cudnn_heuristic64_9.dll' in x[0].lower(),              # 54 MB - heuristics
    # Keep these cuDNN libraries (may be needed for inference):
    # cudnn_cnn64_9.dll                   # 4 MB - core CNN operations
    # cudnn_engines_runtime_compiled64_9.dll # 19 MB - runtime compilation
    # cudnn_graph64_9.dll                 # 2 MB - graph operations  
    # cudnn64_9.dll                       # 0.3 MB - main library
    # Exclude NVRTC alternative DLL
    'nvrtc64_120_0.alt.dll' in x[0].lower(),
    # OPTIONAL: Uncomment these to exclude additional large libraries:
    'cublaslt64_12.dll' in x[0].lower(),    # 638 MB
    'cufft64_11.dll' in x[0].lower(),       # 274 MB  
    'cusolver64_11.dll' in x[0].lower(),    # 270 MB
    'cusolverMg64_11.dll' in x[0].lower(),  # 179 MB
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