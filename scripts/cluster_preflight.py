#!/usr/bin/env python3
"""Scheduled CUDA preflight; never fall back to CPU for model experiments."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback


def main():
    report = {'schema_version': 1, 'status': 'failed', 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
              'array_task_id': os.getenv('SLURM_ARRAY_TASK_ID'), 'python': sys.version,
              'hostname': platform.node(), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
              'model_device_placement': 'pending model-adapter validation'}
    try:
        if not report['slurm_job_id']:
            raise RuntimeError('GPU preflight must run inside a scheduled Slurm allocation')
        import torch
        report.update(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        report.update(device=f'cuda:{device}', gpu_name=props.name, vram_bytes=total,
                      free_vram_bytes=free, bf16_supported=torch.cuda.is_bf16_supported(),
                      capability=list(torch.cuda.get_device_capability(device)))
        dtype_name = os.getenv('REUSE_DTYPE', 'auto')
        if dtype_name not in {'auto', 'float32', 'float16', 'bfloat16'}:
            raise RuntimeError(f'Unsupported REUSE_DTYPE: {dtype_name}')
        if dtype_name == 'bfloat16' and not report['bf16_supported']:
            raise RuntimeError('Requested bfloat16 is unsupported by this GPU/backend')
        if free < float(os.getenv('REUSE_MIN_VRAM_GIB', '0')) * 1024 ** 3:
            raise RuntimeError('Insufficient free GPU memory for the declared requirement')
        dtype = getattr(torch, dtype_name) if dtype_name != 'auto' else torch.float32
        # Test operators used by the audited product hook, without choosing model precision.
        x = torch.randn(4, 8, device=device, dtype=dtype)
        w = torch.randn(8, 8, device=device, dtype=dtype)
        result = (torch.nn.functional.silu(x) * x) @ w
        torch.cuda.synchronize(device)
        if not torch.isfinite(result).all():
            raise RuntimeError('Required GPU operator probe produced nonfinite values')
        signature = {key: report[key] for key in ('torch_version', 'cuda_runtime', 'gpu_name', 'capability')}
        signature['requested_dtype'] = dtype_name
        report.update(status='passed', requested_dtype=dtype_name, operator_probe_device=str(result.device),
                      backend_signature=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest())
        expected = os.getenv('REUSE_EXPECTED_BACKEND', '')
        if expected and report['backend_signature'] != expected:
            report['status'] = 'failed'
            raise RuntimeError('GPU/backend changed since parity validation; rerun validate before research stages')
    except Exception as exc:
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
    root = Path(os.environ.get('REUSE_RUN_ROOT', '.')) / 'preflight'
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{report['slurm_job_id'] or 'unscheduled'}_{report['array_task_id'] or 'single'}.json"
    with tempfile.NamedTemporaryFile('w', dir=root, delete=False) as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, target)
    print(json.dumps(report, sort_keys=True))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    sys.exit(main())
