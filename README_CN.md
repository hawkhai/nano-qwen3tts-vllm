# nano-qwen3tts-vllm

Qwen3-TTS  nano vLLM   optimizations for fast text-to-speech generation.

## 

  Qwen3-TTS  H100  RTF (Real-Time Factor)  "nano-vllm"  1k  

 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)  [Qwen3TTS](https://github.com/QwenLM/Qwen3-TTS)  vLLM 

Qwen3-TTS  [vllm-omni](https://github.com/vllm-project/vllm-omni) 

## 

> [!IMPORTANT]
>  ** 8  1.7B  H100   !

### 

- ** **   batch   prefill/decode  
- **  **  KV cache  block tables  slot mapping   
- **CUDA Graph **   CUDA graph ( batch / decode )  
- ** **  ZMQ  codec chunk  API  PCM   POST /v1/audio/speech  StreamingResponse

##  ( + ZMQ)

  USE_MULTIPROCESS_ENGINES=1   API   talker predictor    ZeroMQ (ZMQ) TCP   main **PUSH**  **PULL**   result-bridge thread   asyncio Future  

```mermaid
flowchart TB
    subgraph Main[""]
        API[FastAPI / StreamingResponse]
        IF[ & ]
        TL[talker_loop_mp]
        PL[predictor_loop_mp]
        TC[TalkerWorkerClient\nPUSH ]
        PC[PredictorWorkerClient\nPUSH ]
        Bridge[Result bridge thread\nPULL  -> Futures]
        API --> IF
        IF --> TL
        IF --> PL
        TL --> TC
        TL --> Bridge
        PL --> PC
        PL --> Bridge
        Bridge --> TL
        Bridge --> PL
    end

    subgraph Talker["Talker "]
        TPULL[PULL\n]
        TLLM[TalkerLLM]
        TPUSH[PUSH\n]
        TPULL --> TLLM --> TPUSH
    end

    subgraph Predictor["Predictor "]
        PPULL[PULL\n]
        PLLM[PredictorLLM]
        PPUSH[PUSH\n]
        PPULL --> PLLM --> PPUSH
    end

    TC -->|"add_request, run_step,\nclear_request (pickle)"| TPULL
    TPUSH -->|"step_id, outputs_all\n(pickle)"| Bridge
    PC -->|" (pickle)"| PPULL
    PPUSH -->|" (pickle)"| Bridge
```

- ** ** (  -> ): `workers/protocol.py` (pickle + numpy)  `add_request` `run_step` `clear_request` `shutdown`
- ** ** (  -> ): worker PUSH  main PULL  result-bridge thread  asyncio Future   `(engine_type, msg_type, payload)`  `request_queues[request_id]`  API 

## 

###  ( )

 NVIDIA H100  `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`  `examples/quick_benchmark.py` :
|  | nano-vllm |  Qwen3-TTS |  |
|--------|-----------|-------------------|-------------|
| ** ** | 2.612s | 8.487s | **3.25x ** |
| ** ** | 0.399 | 1.467 | **3.68x ** |


 NVIDIA L4  `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`  `examples/quick_benchmark.py` :
|  | nano-vllm |  Qwen3-TTS |  |
|--------|-----------|-------------------|-------------|
| ** ** | 4.319s | 16.613s | **3.85x ** |
| ** ** | 0.742 | 3.311 | **4.46x ** |


** :**
-  **4.86x ** 
-  **RTF < 0.4**  nano-vllm   **2.8x **
-   RTF ~2.0 ( )
-  

###  ( )

 L4 GPU 0.6B   decode wav  1 chunk

|  |  (16 codec ) |   | RTF |
|------|--------------------------------------|---------------------|-----|
| **1 CCU** | 160 ms | 50 ms | 0.65 |
| **2 CCUs** | 250 ms | 90 ms | 1.125 |

*(CCU = concurrent request / "concurrent chunk unit" )*

### 
-  ** **  CustomVoice ( ) VoiceDesign ( )  Base ( )
-  ** **  ICL  x_vector_only  
-  ** **    
-  ** **   codec chunk 
-  ** **    

## 

** 

- Python 3.10
- PyTorch 2.10  CUDA
- ** ** 8.0  Ampere/Ada  Flash Attention
- `qwen-tts` `transformers`  

**Flash Attention **

  :

```bash
# : Python 3.12, CUDA 12.4, PyTorch 2.5
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl
```

 Python CUDA PyTorch :  
[https://github.com/mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)

** **

```bash
git clone https://github.com/hawkhai/nano-qwen3tts-vllm.git
cd nano-qwen3tts-vllm
uv sync
# 
pip install -e .

source /data/venv/base/bin/activate
sudo rm -rf /home/yangquanhai/.cache/uv/
cd /data/nano-qwen3tts-vllm
export UV_CACHE_DIR=/data/uvcache/.cache/uv
uv sync --index-url https://mirrors.aliyun.com/pypi/simple

```

## 

nano-qwen3tts-vllm  ** Qwen3-TTS **:

|  |  |   |  |  |
|---|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign |    |    |  |  |
| Qwen3-TTS-12Hz-1.7B-CustomVoice |       9      |    |  |  |
| Qwen3-TTS-12Hz-1.7B-Base |  3      |    |  |  |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 9      |    |  |  |
| Qwen3-TTS-12Hz-0.6B-Base |  3      |    |  |  |

 **12Hz** ( )  **25Hz** () 

>  ** **  [`examples/`](examples/)   

## API 

**  codec chunk :**

1. ** codec chunk **  `generate_*()`  `list()`   
2. ** **  `interface.speech_tokenizer.decode()`  

:
-   codec   
-  API  
-    
-   

## 

### 1.  ( )

 ( Vivian Mike Sarah ):

```python
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
import soundfile as sf

#  CustomVoice 
interface = Qwen3TTSInterface.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    enforce_eager=False,
    tensor_parallel_size=1,
)

#  codec chunk
audio_codes = list(interface.generate_custom_voice(
    text="Hello, this is a test.",
    language="English",
    speaker="Vivian",
))

#   speech tokenizer 
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
sf.write("output.wav", wavs[0], sr)
```

** :** Vivian Mike Sarah Laura Alex Ethan Emma  

### 2.  ( )

 :

```python
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
import soundfile as sf

#  VoiceDesign 
interface = Qwen3TTSInterface.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    enforce_eager=False,
    tensor_parallel_size=1,
)

#   
audio_codes = list(interface.generate_voice_design(
    text="Hi! How are you doing today?",
    language="English",
    instruct="A young woman with a warm, friendly voice and slight excitement",
))

#   speech tokenizer 
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
sf.write("output_designed.wav", wavs[0], sr)
```

** :** [`examples/voice_design_example.py`](examples/voice_design_example.py)  

### 3.  ( )

 :

```python
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
import soundfile as sf

#  Base 
interface = Qwen3TTSInterface.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    enforce_eager=False,
    tensor_parallel_size=1,
)

#  
ref_audio, ref_sr = sf.read("reference.wav")

#    (ICL  -  )
voice_clone_prompt = interface.create_voice_clone_prompt(
    ref_audio=(ref_audio, ref_sr),
    ref_text="This is the reference text that was spoken in the audio.",
    x_vector_only_mode=False,  # ICL   
)

#   
audio_codes = list(interface.generate_voice_clone(
    text="Hello, this is a cloned voice speaking.",
    language="English",
    voice_clone_prompt=voice_clone_prompt,
))

#   speech tokenizer 
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
sf.write("output_cloned.wav", wavs[0], sr)
```

** :**
- **ICL ** (`x_vector_only_mode=False`):      `ref_text`       
- **x_vector_only ** (`x_vector_only_mode=True`):     `ref_text`     

** :** [`examples/voice_clone_example.py`](examples/voice_clone_example.py)  ICL  x_vector  

###  ( )

  talker + predictor    codec chunk  `POST /v1/audio/speech`  PCM  

```python
# :  PCM   examples/client.py
import requests
r = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"text": "Hello world.", "language": "English", "speaker": "Vivian"},
    stream=True,
)
#  r.iter_content()   WAV
```

### 

```bash
export QWEN3_TTS_MODEL_PATH=/path/to/qwen3tts
python -m uvicorn examples.server:app --host 0.0.0.0 --port 8000
# 
python examples/server.py
```

## 

|  |  |
|-----------|--------------|
| `model_path` | Qwen3-TTS  () |
| `enforce_eager` |  CUDA graph () |
| `tensor_parallel_size` | GPU  (1-8) |
| (multiprocess) | Talker predictor    () |
| `QWEN3_TTS_MODEL_PATH` |   () |

## 

>  ** [`examples/`](examples/)   !**

 [`examples/`](examples/) :

- **[`custom_voice_example.py`](examples/custom_voice_example.py)** -   
- **[`voice_design_example.py`](examples/voice_design_example.py)** -    
- **[`voice_clone_example.py`](examples/voice_clone_example.py)** -  ICL  x_vector  
- **[`server.py`](examples/server.py)** - FastAPI    
- **[`client.py`](examples/client.py)** -   API

### 

```bash
# 
python examples/custom_voice_example.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --text "Hello world" \
    --speaker Vivian

# 
python examples/voice_design_example.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --output-dir output

# 
python examples/voice_clone_example.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --ref-audio reference.wav \
    --ref-text "Reference transcript" \
    --output-dir output
```

## 

### 

 :

```python
interface = Qwen3TTSInterface(
    model_path="/path/to/local/model",
    enforce_eager=False,
    tensor_parallel_size=1,
)
```

### 

 :

```python
audio_codes = list(interface.generate_voice_design(
    text="Hello world",
    language="English",
    instruct="professional voice",
    non_streaming_mode=True,  #   
))
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
```

### 

 :

- ** ** -   
- ** ** ( ) -    
- ** ** -   

 :

```python
audio_codes = list(interface.generate_voice_design(
    text="",
    language="Chinese",
    instruct="",
))
wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
```

## 

1. **CUDA Graph** (`enforce_eager=False`)  2-3x 
2. **12Hz **   (25Hz )
3. ** ** (ZMQ)    
4. ** **  codec chunk   

##  ( )

-   (CustomVoice VoiceDesign Base)
-  ICL  x_vector  
-    
-   CUDA Graph

## Star History
![Star History Chart](https://api.star-history.com/svg?repos=tsdocode/nano-qwen3tts-vllm&type=Date?l=1)
