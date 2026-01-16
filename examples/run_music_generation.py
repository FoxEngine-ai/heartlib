from heartlib import HeartMuLaGenPipeline
import argparse
import torch
from transformers import BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--version", type=str, default="3B")
    parser.add_argument("--lyrics", type=str, default="./assets/lyrics.txt")
    parser.add_argument("--tags", type=str, default="./assets/tags.txt")
    parser.add_argument("--save_path", type=str, default="./assets/output.mp3")

    parser.add_argument("--max_audio_length_ms", type=int, default=240_000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--bnb", type=str, choices=["4bit", "8bit"], default=None,
                        help="Enable bitsandbytes quantization (4bit or 8bit)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for faster inference")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default caches better, reduce-overhead faster)")
    parser.add_argument("--compile-cache-dir", type=str, default=None,
                        help="Directory to cache compiled models (avoids recompilation)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    bnb_config = None
    if args.bnb == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif args.bnb == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        version=args.version,
        bnb_config=bnb_config,
        compile=args.compile,
        compile_mode=args.compile_mode,
        compile_cache_dir=args.compile_cache_dir,
    )
    with torch.no_grad():
        pipe(
            {
                "lyrics": args.lyrics,
                "tags": args.tags,
            },
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.save_path,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
    print(f"Generated music saved to {args.save_path}")
