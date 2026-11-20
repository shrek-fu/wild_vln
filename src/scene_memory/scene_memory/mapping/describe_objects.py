"""DAM (Describe Anything, ``nvidia/DAM-3B``) captions for saved object key views.

After mapping (one shot; also merges text into ``objects_4d.json``)::

    python3 -m scene_memory.mapping.describe_objects data/scene4d/latest

During mapping, as a background worker next to ``hydra_scene_node`` (only
writes ``descriptions.json``; the node merges it on every export)::

    python3 -m scene_memory.mapping.describe_objects <scene_dir> --watch 5

``--precision auto`` loads the LLM in 4-bit (~3.4 GB GPU) unless there is room
for fp16 (~7 GB).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from scene_memory import paths
from scene_memory.mapping.object_views import descriptions_by_key, load_description_cache, save_description_cache

DAM_ROOT = paths.REFER / "describe-anything"
DEFAULT_CKPT = paths.DAM
DEFAULT_QUERY = "<image>\nDescribe the masked region in detail."
FP16_MIB = 7500
INT4_MIB = 3800


def dam_prompt_mode(mode: str) -> str:
    """``focal_prompt`` → ``full+focal_crop`` (the alias the DAM demos and DAAAM use)."""
    return f"full+{mode.replace('_prompt', '_crop')}" if mode.endswith("_prompt") else mode


def load_dam(model_path: str, conv_mode: str, prompt_mode: str, precision: str):
    root = str(DAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    from dam import DescribeAnythingModel, disable_torch_init

    disable_torch_init()
    kwargs = {}
    if precision == "4bit":
        from transformers import BitsAndBytesConfig

        # DAM's own load_4bit passes load_in_4bit and quantization_config together,
        # which transformers>=4.4x rejects; torch_dtype is still required by its config prep.
        kwargs = {
            "torch_dtype": torch.float16,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        }
    model = DescribeAnythingModel(
        model_path=model_path,
        conv_mode=conv_mode,
        prompt_mode=dam_prompt_mode(prompt_mode),
        **kwargs,
    )
    if precision == "4bit":
        return model
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def gpu_free_mib() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return int(free / 2**20)
    except Exception:  # noqa: BLE001
        return None


def pick_precision(requested: str) -> str:
    if requested != "auto":
        return requested
    free = gpu_free_mib()
    return "fp16" if free is not None and free >= FP16_MIB else "4bit"


def pending_views(scene_dir: Path, cache: dict, force: bool) -> list[tuple[dict, dict]]:
    memory_path = scene_dir / "objects_4d.json"
    if not memory_path.is_file():
        return []
    try:
        scene = json.loads(memory_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    todo = []
    for obj in scene.get("objects", []):
        for view in obj.get("views", []):
            cached = cache.get(view["image"])
            # A view file is overwritten in place when a better frame arrives.
            if force or cached is None or abs(float(cached["stamp"]) - float(view["stamp"])) > 1e-6:
                if (scene_dir / view["image"]).is_file() and (scene_dir / view["mask"]).is_file():
                    todo.append((obj, view))
    return todo


def describe_view(dam, scene_dir: Path, view: dict, args) -> str | None:
    from PIL import Image

    image = Image.open(scene_dir / view["image"]).convert("RGB")
    mask_np = (np.asarray(Image.open(scene_dir / view["mask"]).convert("L")) > 127).astype(np.uint8) * 255
    if not mask_np.any():
        return None
    text = dam.get_description(
        image,
        Image.fromarray(mask_np),
        args.query,
        streaming=False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
    )
    return str(text).strip()


def merge_into_memory(scene_dir: Path, cache: dict) -> None:
    memory_path = scene_dir / "objects_4d.json"
    scene = json.loads(memory_path.read_text(encoding="utf-8"))
    by_key = descriptions_by_key(cache)
    for obj in scene.get("objects", []):
        obj["descriptions"] = by_key.get(obj["key"], [])
    tmp = memory_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(scene, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(memory_path)
    described = sum(1 for obj in scene.get("objects", []) if obj["descriptions"])
    print(f"merged descriptions for {described}/{len(scene.get('objects', []))} objects into {memory_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="scene4d directory (contains objects_4d.json and views/)")
    parser.add_argument(
        "--model_path",
        default=str(DEFAULT_CKPT if DEFAULT_CKPT.is_dir() else "nvidia/DAM-3B"),
        help="Local DAM-3B dir (models/DAM-3B) or Hugging Face id nvidia/DAM-3B",
    )
    parser.add_argument("--precision", default="auto", choices=("auto", "fp16", "4bit"))
    parser.add_argument("--conv_mode", default="v1")
    parser.add_argument("--prompt_mode", default="focal_prompt")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="Re-describe views that already have text")
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Keep running and poll for new key views every N seconds (live mapping); 0 = one shot",
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene)
    precision = pick_precision(args.precision)
    need = FP16_MIB if precision == "fp16" else INT4_MIB
    dam = None
    force = args.force
    while True:
        # ``latest`` may be re-pointed by a new run, so resolve every round.
        resolved = scene_dir.resolve()
        cache = load_description_cache(resolved)
        todo = pending_views(resolved, cache, force)
        force = False
        if todo and dam is None:
            free = gpu_free_mib()
            if free is not None and free < need:
                message = f"only {free} MiB GPU free, DAM-3B {precision} needs ~{need} MiB"
                if not args.watch:
                    raise SystemExit(message + "; stop the live pipeline or use --precision 4bit")
                print(message + "; waiting", flush=True)
                time.sleep(max(args.watch, 5.0))
                continue
            started = time.perf_counter()
            dam = load_dam(args.model_path, args.conv_mode, args.prompt_mode, precision)
            print(f"DAM-3B ({precision}) loaded in {time.perf_counter() - started:.1f}s from {args.model_path}", flush=True)
        if todo:
            print(f"{len(todo)} views to describe ({len(cache)} cached) in {resolved}", flush=True)
        for index, (obj, view) in enumerate(todo, 1):
            t0 = time.perf_counter()
            text = describe_view(dam, resolved, view, args)
            if text is None:
                continue
            cache[view["image"]] = {
                "key": obj["key"],
                "name": obj["name"],
                "visit": view["visit"],
                "stamp": view["stamp"],
                "text": text,
                "model": Path(args.model_path).name,
                "precision": precision,
            }
            save_description_cache(resolved, cache)
            print(
                f"[{index}/{len(todo)}] {obj['key']} {obj['name']} v{view['visit']} "
                f"({time.perf_counter() - t0:.1f}s): {text}",
                flush=True,
            )
        if not args.watch:
            if (resolved / "objects_4d.json").is_file():
                merge_into_memory(resolved, cache)
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
