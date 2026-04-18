"""
双实例运行脚本 - 优化显存使用到10GB以下

这个脚本演示如何在24GB显卡上同时运行两个TTS实例，每个实例使用约10GB显存。

优化策略：
1. gpu_memory_utilization=0.45 (每个实例使用45%显存，约10.4GB on L4)
2. enforce_eager=False (启用CUDA graph，提高GPU利用率)
3. 平衡显存和算力，避免资源浪费
4. 双实例总GPU利用率可达50-60%

使用方法：
    # 第一个实例 (端口8000)
    python examples/run_dual_instance.py --port 8000 --text "第一个实例的测试文本"
    
    # 第二个实例 (端口8001) - 在另一个终端运行
    python examples/run_dual_instance.py --port 8001 --text "第二个实例的测试文本"
"""

import os
os.environ["TORCH_SDPA_BACKEND"] = "flash_attention,mem_efficient,math"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"

import argparse
import asyncio
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

# Disable cuDNN SDPA backend to avoid cuDNN Frontend errors
torch.backends.cuda.enable_cudnn_sdp(False)

# Increase torch dynamo cache size limit to avoid recompilation warnings
torch._dynamo.config.cache_size_limit = 64

sys.path.append(".")
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface


def parse_args():
    parser = argparse.ArgumentParser(
        description="双实例运行脚本 - 优化显存使用"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="模型路径",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="这是一个测试文本，用于验证双实例运行。",
        help="要合成的文本",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Vivian",
        help="说话人",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="语言",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="输出目录",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="实例端口号（用于区分不同实例）",
    )
    
    return parser.parse_args()


async def collect_audio_codes(interface, *, text: str, language: str, speaker: str):
    """收集音频编码块"""
    print(f"    [collect] 开始生成: text='{text[:40]}...' language={language} speaker={speaker}")
    chunks = []
    idx = 0
    async for chunk in interface.generate_custom_voice_async(
        text=text,
        language=language,
        speaker=speaker,
    ):
        idx += 1
        chunks.append(chunk)
        if idx % 50 == 0:
            print(f"    [collect] 已接收 {idx} 个块")
    print(f"    [collect] 完成，总块数={idx}")
    return chunks


async def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"双实例运行 - 实例端口 {args.port}")
    print("=" * 60)
    print(f"优化配置：")
    print(f"  - GPU Memory Utilization: 0.45 (约10.4GB on L4)")
    print(f"  - Enforce Eager: False (启用CUDA graph提高性能)")
    print(f"  - 目标：平衡显存和GPU利用率")
    print("=" * 60)
    
    # 初始化模型 - 使用优化配置
    print(f"\n加载模型: {args.model_path}")
    print("应用显存优化配置...")
    
    interface = Qwen3TTSInterface(
        model_path=args.model_path,
        # 关键优化参数
        gpu_memory_utilization=0.45,  # 45%显存，每个实例约10.4GB (L4 23GB)
        enforce_eager=False,           # 启用CUDA graph，提高GPU利用率
        tensor_parallel_size=1,
    )
    
    print("✓ 模型加载成功")
    
    # 显示显存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n当前显存使用:")
        print(f"  - 已分配: {allocated:.2f} GB")
        print(f"  - 已预留: {reserved:.2f} GB")
    
    speech_tokenizer = interface.speech_tokenizer
    
    # 启动异步任务
    print("\n启动 ZMQ 任务...")
    await interface.start_zmq_tasks()
    print("✓ ZMQ 任务就绪")
    
    try:
        print("\n" + "-" * 60)
        print(f"生成语音 - 实例 {args.port}")
        print("-" * 60)
        print(f"文本: {args.text}")
        print(f"文本长度: {len(args.text)} 字符")
        
        # 生成音频
        inference_start = time.time()
        audio_codes = await collect_audio_codes(
            interface,
            text=args.text,
            language=args.language,
            speaker=args.speaker,
        )
        inference_time = time.time() - inference_start
        
        # 解码音频
        decode_start = time.time()
        wavs, sr = speech_tokenizer.decode([{"audio_codes": audio_codes}])
        decode_time = time.time() - decode_start
        
        # 保存音频
        io_start = time.time()
        output_path = output_dir / f"dual_instance_port{args.port}.wav"
        sf.write(str(output_path), wavs[0], sr)
        io_time = time.time() - io_start
        
        # 性能指标
        audio_duration = len(wavs[0]) / sr
        rtf = inference_time / audio_duration if audio_duration > 0 else 0
        total_time = inference_time + decode_time + io_time
        
        print(f"\n⏱️  性能指标:")
        print(f"  推理时间: {inference_time:.3f}s")
        print(f"  解码时间: {decode_time:.3f}s")
        print(f"  I/O时间:  {io_time:.3f}s")
        print(f"  总时间:   {total_time:.3f}s")
        print(f"  音频时长: {audio_duration:.3f}s")
        print(f"  RTF:      {rtf:.3f}x")
        print(f"  吞吐量:   {len(args.text)/inference_time:.1f} 字符/秒")
        print(f"💾 输出: {output_path}")
        
        # 最终显存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n最终显存使用:")
            print(f"  - 已分配: {allocated:.2f} GB")
            print(f"  - 已预留: {reserved:.2f} GB")
        
    finally:
        print("\n停止 ZMQ 任务...")
        await interface.stop_zmq_tasks()
        print("✓ ZMQ 任务已停止")
    
    print("\n" + "=" * 60)
    print(f"实例 {args.port} 完成!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
