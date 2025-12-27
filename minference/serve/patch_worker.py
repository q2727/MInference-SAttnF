"""
Patch Model Worker for FastChat.
A generic worker that loads a model and applies custom patches (like MInference, SAttnF, etc.) 
before registering with the FastChat controller.
"""

import argparse
import sys
import uuid
import torch
import uvicorn
import fastchat.serve.model_worker
from fastchat.serve.model_worker import (
    ModelWorker,
    app,
    logger,
)
import fastchat.model
import fastchat.model.model_adapter

# 保存原始加载函数 (从源头保存)
_original_load_model = fastchat.model.model_adapter.load_model

def patched_load_model(model_path, *args, **kwargs):
    """拦截加载过程并应用补丁"""
    print(f"[PatchWorker] Intercepted load_model for {model_path}!")
    
    # 1. 调用原始 FastChat 加载逻辑
    model, tokenizer = _original_load_model(model_path, *args, **kwargs)
    
    # 2. 获取补丁配置 (从全局注入的变量中读取)
    patch_cfg = getattr(fastchat.model, "custom_patch_args", {})
    
    if patch_cfg.get("enabled", False):
        patch_type = patch_cfg.get("patch_type", "minference")
        print(f"[PatchWorker] Applying custom patch: {patch_type}...")
        
        try:
            if patch_type in ["minference", "sattnf", "flexprefill", "xattention"]:
                from minference import MInference
                # MInference 封装了这些算法的接入逻辑
                patch_manager = MInference(
                    attn_type=patch_type,
                    model_name=model_path,
                    kv_type=patch_cfg.get("kv_type", "dense"),
                    config_path=patch_cfg.get("config_path", None)
                )
                model = patch_manager(model)
                print(f"[PatchWorker] Successfully applied {patch_type} patch.")
            else:
                print(f"[PatchWorker] Warning: Unknown patch type: {patch_type}, skipping.")
        except Exception as e:
            print(f"[PatchWorker] Error: Failed to apply patch {patch_type}: {e}")
            raise e
            
    return model, tokenizer

# 强力替换：替换源头和各个引用点
fastchat.model.model_adapter.load_model = patched_load_model
fastchat.model.load_model = patched_load_model
fastchat.serve.model_worker.load_model = patched_load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # FastChat 标准参数
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-names", type=lambda s: s.split(","), help="Optional display names")
    parser.add_argument("--limit-worker-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str, default="20GiB")
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--cpu-offloading", action="store_true")
    
    # 补丁相关参数 (通用名)
    parser.add_argument("--enable-patch", action="store_true", help="Enable custom model patching")
    parser.add_argument("--patch-type", type=str, default="minference", 
                        help="Type of patch to apply (e.g., minference, sattnf, flexprefill)")
    parser.add_argument("--kv-type", type=str, default="dense", help="KV cache type")
    parser.add_argument("--config-path", type=str, default=None, help="Path to offline config/pattern")

    args = parser.parse_args()

    # 类型转换
    if args.dtype == "fp16":
        args.dtype = torch.float16
    elif args.dtype == "bf16":
        args.dtype = torch.bfloat16
    elif args.dtype == "fp32":
        args.dtype = torch.float32

    # 将参数注入到 fastchat.model 模块供 patched_load_model 使用
    fastchat.model.custom_patch_args = {
        "enabled": args.enable_patch,
        "patch_type": args.patch_type,
        "kv_type": args.kv_type,
        "config_path": args.config_path
    }

    # 生成 worker_id
    worker_id = str(uuid.uuid4())[:8]

    # 直接实例化 ModelWorker
    worker = ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_path=args.model_path,
        model_names=args.model_names,
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        dtype=args.dtype,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
    )
    
    # 确保全局 worker 被设置，以便 API 路由能找到
    fastchat.serve.model_worker.worker = worker
    
    print(f"Starting {args.patch_type if args.enable_patch else 'standard'} worker at {args.worker_address} (id: {worker_id})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")