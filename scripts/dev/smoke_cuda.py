#!/usr/bin/env python3
"""
CUDA Smoke Test for Speculative Decoding Pipeline
Quick test to verify CUDA functionality and fp16 dtype selection.
"""

import sys
import pathlib
import torch

# Add src to path
sys.path.insert(0, str(pathlib.Path("src").resolve()))

from specdec.pipeline import SpeculativePipeline

def main():
    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| Dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA smoke test")
        return
    
    try:
        pipe = SpeculativePipeline(
            base_model="gpt2", 
            draft_model="distilgpt2", 
            max_draft=2, 
            implementation="hf", 
            enable_optimization=True, 
            device="cuda"
        )
        
        res = pipe.generate(
            "Say hi in 1 sentence.", 
            max_tokens=8, 
            temperature=0.7, 
            do_sample=True
        )
        
        print("CUDA sanity OK. Text:", res["text"])
        print("Device:", res.get("device", "unknown"))
        print("Dtype:", res.get("dtype", "unknown"))
        
    except Exception as e:
        print(f"CUDA smoke test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
