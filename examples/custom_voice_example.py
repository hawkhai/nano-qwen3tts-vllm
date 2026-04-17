"""
Example demonstrating Custom Voice feature.

Custom Voice allows you to generate speech with pre-defined speaker voices
using the CustomVoice model (e.g., Qwen3-TTS-12Hz-1.7B-CustomVoice).

Usage:
    # Using HuggingFace model ID (automatically downloads if needed)
    python examples/custom_voice_example.py \
        --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --text "Hello world" \
        --speaker Vivian
    
    # Using local model path
    python examples/custom_voice_example.py \
        --model-path /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --text "Hello world" \
        --speaker Vivian \
        --output-dir ./output
    
    # Batch processing with multiple texts
    python examples/custom_voice_example.py \
        --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --batch-texts "Hello world" "How are you?" "This is a test" \
        --speaker Vivian \
        --output-dir ./output
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import soundfile as sf

sys.path.append(".")
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface


REQUIRED_MODEL_DOWNLOADS = [
    (
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "Qwen3-TTS-Tokenizer-12Hz",
    ),
    (
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen3-TTS-12Hz-0.6B-Base",
    ),
    (
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ),
]

DEFAULT_CUSTOM_VOICE_DIR = REQUIRED_MODEL_DOWNLOADS[-1][1]


def ensure_required_models(base_dir: Path) -> Optional[Path]:
    """Download required models via ModelScope if they are missing locally."""

    local_custom_voice_dir = base_dir / DEFAULT_CUSTOM_VOICE_DIR

    for model_id, relative_dir in REQUIRED_MODEL_DOWNLOADS:
        target_dir = base_dir / relative_dir
        if target_dir.is_dir() and any(target_dir.iterdir()):
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "modelscope",
            "download",
            "--model",
            model_id,
            "--local_dir",
            str(target_dir),
        ]

        print(f"Missing model '{model_id}'. Downloading to {target_dir} via ModelScope...")
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Cannot find `modelscope` CLI. Install it via `pip install modelscope`."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "ModelScope download failed. Ensure `modelscope download` works manually "
                "and that this machine can access the internet."
            ) from exc

    if local_custom_voice_dir.is_dir() and any(local_custom_voice_dir.iterdir()):
        return local_custom_voice_dir
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Example demonstrating Custom Voice feature"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to CustomVoice model or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice or /path/to/local/model)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the custom voice feature.",
        help="Text to synthesize (single text mode)",
    )
    parser.add_argument(
        "--batch-texts",
        type=str,
        nargs="+",
        default=None,
        help="Multiple texts to synthesize (batch mode)",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Vivian",
        help="Speaker name (default: Vivian)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language (default: English)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for generated audio files",
    )
    parser.add_argument(
        "--prefer-modelscope-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When provided model path is not a local directory, prefer using the locally "
            "downloaded ModelScope assets (if available). Pass --no-prefer-modelscope-local "
            "to disable this behavior."
        ),
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    print("=" * 60)
    print("Custom Voice Example")
    print("=" * 60)

    # Ensure required local assets exist before model initialization
    repo_root = Path(__file__).resolve().parents[1]
    local_modelscope_dir = ensure_required_models(repo_root)

    model_path_to_use = args.model_path
    if (
        args.prefer_modelscope_local
        and local_modelscope_dir is not None
        and not os.path.isdir(args.model_path)
        and not os.path.isfile(args.model_path)
    ):
        model_path_to_use = str(local_modelscope_dir)
        print(
            "Using locally downloaded ModelScope model at: "
            f"{local_modelscope_dir}"
        )

    # Initialize CustomVoice model
    print(f"\nLoading CustomVoice model from: {model_path_to_use}")
    
    # Check if it's a HuggingFace model ID or local path
    # Use from_pretrained which handles both cases automatically
    if os.path.isdir(model_path_to_use) or os.path.isfile(model_path_to_use):
        # Local path - use regular init
        interface = Qwen3TTSInterface(
            model_path=model_path_to_use,
            enforce_eager=False,
            tensor_parallel_size=1,
        )
    else:
        # Likely a HuggingFace model ID, use from_pretrained
        print("  Detected HuggingFace model ID, downloading if needed...")
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=model_path_to_use,
            enforce_eager=False,
            tensor_parallel_size=1,
        )
    print("Model loaded successfully")
    
    # Use the interface's built-in speech tokenizer for decoding
    speech_tokenizer = interface.speech_tokenizer
    
    # Determine if we're doing single text or batch processing
    if args.batch_texts:
        texts = args.batch_texts
        mode = "batch"
    else:
        texts = [args.text]
        mode = "single"
    
    print(f"\nMode: {mode} processing")
    print(f"Speaker: {args.speaker}")
    print(f"Language: {args.language}")
    print(f"Number of texts: {len(texts)}")
    
    # Example 1: Single text generation (or batch processing)
    print("\n" + "-" * 60)
    print(f"[{mode.title()}] Generating speech with custom voice")
    print("-" * 60)
    
    total_start_time = time.time()
    
    for i, text in enumerate(texts, 1):
        print(f"\n[{i}/{len(texts)}] Text: {text}")
        start_time = time.time()
        
        # Generate codec chunks
        audio_codes = list(interface.generate_custom_voice(
            text=text,
            language=args.language,
            speaker=args.speaker,
        ))
        
        # Decode to audio
        wavs, sr = speech_tokenizer.decode([{"audio_codes": audio_codes}])
        
        elapsed = time.time() - start_time
        
        # Generate output filename
        if mode == "single":
            output_path = output_dir / "custom_voice_output.wav"
        else:
            output_path = output_dir / f"custom_voice_batch_{i}.wav"
        
        sf.write(str(output_path), wavs[0], sr)
        print(f"  Generated in {elapsed:.2f}s")
        print(f"  Audio duration: {len(wavs[0])/sr:.2f}s")
        print(f"  Saved to: {output_path}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal processing time: {total_elapsed:.2f}s")
    
    # Example 2: Different speakers (if in single mode)
    if mode == "single":
        print("\n" + "-" * 60)
        print("[Example 2] Different speakers demonstration")
        print("-" * 60)
        
        # Common available speakers for CustomVoice models
        speakers = ["Vivian", "Mike", "Sarah", "Laura", "Alex", "Ethan", "Emma"]
        demo_text = "Hello, this is a demonstration of different speaker voices."
        
        # Test a few different speakers
        test_speakers = speakers[:3]  # Test first 3 speakers
        
        for speaker in test_speakers:
            print(f"\nSpeaker: {speaker}")
            print(f"  Text: {demo_text}")
            
            start_time = time.time()
            audio_codes = list(interface.generate_custom_voice(
                text=demo_text,
                language=args.language,
                speaker=speaker,
            ))
            wavs, sr = speech_tokenizer.decode([{"audio_codes": audio_codes}])
            elapsed = time.time() - start_time
            
            output_path = output_dir / f"custom_voice_{speaker.lower()}.wav"
            sf.write(str(output_path), wavs[0], sr)
            print(f"  Generated in {elapsed:.2f}s, saved to: {output_path}")
    
    # Example 3: Different languages (if in single mode)
    if mode == "single":
        print("\n" + "-" * 60)
        print("[Example 3] Multi-language demonstration")
        print("-" * 60)
        
        multilingual_examples = [
            {
                "text": "Hello, this is English speech synthesis.",
                "language": "English",
                "output": "custom_voice_english.wav",
            },
            {
                "text": "Hola, esto es síntesis de voz en español.",
                "language": "Spanish",
                "output": "custom_voice_spanish.wav",
            },
            {
                "text": "Bonjour, ceci est une synthèse vocale en français.",
                "language": "French",
                "output": "custom_voice_french.wav",
            },
            {
                "text": "Guten Tag, dies ist Sprachsynthese auf Deutsch.",
                "language": "German",
                "output": "custom_voice_german.wav",
            },
            {
                "text": "Ciao, questo è un sintesi vocale in italiano.",
                "language": "Italian",
                "output": "custom_voice_italian.wav",
            },
        ]
        
        for example in multilingual_examples:
            print(f"\nLanguage: {example['language']}")
            print(f"  Text: {example['text']}")
            
            start_time = time.time()
            audio_codes = list(interface.generate_custom_voice(
                text=example["text"],
                language=example["language"],
                speaker=args.speaker,
            ))
            wavs, sr = speech_tokenizer.decode([{"audio_codes": audio_codes}])
            elapsed = time.time() - start_time
            
            output_path = output_dir / example["output"]
            sf.write(str(output_path), wavs[0], sr)
            print(f"  Generated in {elapsed:.2f}s, saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print(f"\nAll audio files saved to: {output_dir.absolute()}")
    
    # Print available speakers info
    print(f"\nAvailable speakers for CustomVoice models:")
    print("  - Vivian (Female, adult)")
    print("  - Mike (Male, adult)")
    print("  - Sarah (Female, adult)")
    print("  - Laura (Female, adult)")
    print("  - Alex (Male, young adult)")
    print("  - Ethan (Male, young adult)")
    print("  - Emma (Female, young adult)")
    print("  - And more (check model documentation for complete list)")


if __name__ == "__main__":
    main()
