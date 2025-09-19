

Windows setup meant for use with SkyrimNet either locally (or local secondary PC) install using coqui-ai-TTS using either the "XTTS" or "Zonos" endpoints. 
- should support Blackwell cards
- model files in `models` folder
- latents in `latents_pt` folder by language type (no json support)
- wav files in speakers/[language] will be converted once on startup
- output files saved in `output_temp` folder under process timestamp folders
- default server is at http://localhost:7860
- Gradio UI is available there also.

Based on [coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS)

Assumes that [Python 3.12](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) is already installed. 

To install other needed files:

`1_Install.bat` 

To run:

`2_Start.bat` 

This should start in a high priority process window.

`2_Start_DeepSpeed.bat` (for GPU support with DeepSpeed)
or
`2_Start_CPU.bat` (for CPU only)

Be sure to set the Language section in the SkyrimNet Zonos tab to match the language you want.

SUPPORTED_LANGUAGES =  ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"]

Note:
Currently hardcoded as the default values in the SkyrimNet Zonos configuration dont work well.

    speed=1.0,  # speaking_rate if speaking_rate else 1.0,
    top_p=1.0,  # top_p if top_p else 1.0,
    top_k=50,   # top_k if top_k else 50,
    temperature=0.7,

remove the bit like (1.0,  #) in lines 113-117 in skyrimnet-xtts.py and then they can be controlled in the SkyrimNet UI

---

To run by hand:

py -3.12 -m venv .venv
.venv\scripts\activate
pip install -r requirements.txt

python -m skyrimnet-xtts
