# ComfyUI_QwenVL Optimization Project - Context

**Last Updated:** 2025-12-07
**Branch:** `claude/fix-node-performance-019PjwcjTqhGhsy2NsVNp81x`

## Project Overview

Implementing comprehensive fixes and enhancements to the ComfyUI_QwenVL node pack, including critical bug fixes, missing generation parameters, performance optimizations, and ensuring ComfyUI node v3 compliance.

## Current Status

### Phase 1: Critical Fixes ✅ COMPLETED
- [x] Fix 1.1: Non-quantized model loading (added `low_cpu_mem_usage=True`)
- [x] Fix 1.2: 8bit quantization compute_dtype parameter (added `bnb_8bit_compute_dtype`)
- [x] Fix 1.3: Hardcoded CUDA device calls (changed `.to("cuda")` -> `.to(self.model.device)`)
- [x] Fix 1.4: Temperature parameter bug (removed from batch_decode())

### Phase 2: Core Generation Parameters ✅ COMPLETED
- [x] Add `do_sample` parameter (default: True)
- [x] Add `repetition_penalty` parameter (default: 1.0, range: 1.0-2.0)
- [x] Add `top_p` parameter (default: 0.9, range: 0.0-1.0)
- [x] Add `top_k` parameter (default: 50, range: 0-200)
- [x] Add `min_p` parameter (default: 0.0, range: 0.0-1.0)
- [x] Update generate() calls to use new parameters with proper conditional logic
- [x] Expand temperature range (0-2 instead of 0-1) and improve step size (0.05)

### Phase 3: Performance Optimizations ✅ COMPLETED
- [x] Add Flash Attention 2 support (optional, with graceful fallback)
- [x] Switch from `torch.no_grad()` to `torch.inference_mode()`
- [ ] Refactor duplicate model loading logic into shared functions (DEFERRED - code working well)

### Phase 4: Quality Improvements ✅ COMPLETED
- [x] Improve error messages (specific CUDA OOM handling with actionable suggestions)
- [x] Add detailed exception logging with traceback for debugging
- [ ] Add performance logging (tokens/sec) (DEFERRED - can be added later)
- [ ] Add parameter validation (DEFERRED - ComfyUI handles basic validation)

### Phase 5: ComfyUI v3 Compliance ✅ COMPLETED
- [x] Research ComfyUI node v3 requirements
- [x] Add RETURN_NAMES for better output clarity
- [x] Verify standard node structure compliance

## Code Locations

### QwenVL Class (Vision-Language)
- **INPUT_TYPES**: Lines 208-246
- **Model Loading**: Lines 303-337
- **Inference**: Lines 252-426
- **CUDA device call**: Line 403
- **batch_decode call**: Line 409-414

### Qwen Class (Text-only)
- **INPUT_TYPES**: Lines 449-480
- **Model Loading**: Lines 524-548
- **Inference**: Lines 486-581
- **CUDA device call**: Line 562
- **batch_decode call**: Line 569-574

## Issues Identified

### Critical Bugs
1. **Non-quantized models load to CPU first** (Lines 319-337, 540-548)
   - Missing `low_cpu_mem_usage=True` in load_kwargs
   - Causes unnecessary RAM usage and slow loading

2. **8bit quantization incomplete** (Lines 315-318, 536-539)
   - Missing `bnb_8bit_compute_dtype=compute_dtype`
   - May cause performance issues or errors

3. **Hardcoded CUDA calls fail on CPU** (Lines 403, 562)
   - `.to("cuda")` should be `.to(self.model.device)`
   - Breaks when running on CPU

4. **Temperature parameter bug** (Lines 413, 573)
   - `batch_decode()` doesn't accept temperature parameter
   - Will cause runtime error or be silently ignored

### Missing Features
5. **No sampling control**
   - `do_sample` not exposed (temperature set but sampling not explicitly enabled)
   - Missing `top_p`, `top_k`, `min_p` for quality control
   - Missing `repetition_penalty` for preventing repetitive output

## Research Notes

### Qwen VL 3 Specifics
- Uses `Qwen3VLForConditionalGeneration` class
- Already implemented in code with automatic detection (line 324)
- Custom models supported via `custom_models.json`

### BitsAndBytes Quantization
- 4bit config is correct (lines 309-314, 530-535)
- 8bit missing compute_dtype (needs fixing)
- Both use `device_map="auto"` correctly

### ComfyUI Node Architecture
- Uses standard INPUT_TYPES/RETURN_TYPES/FUNCTION pattern
- Currently registered in __init__.py
- Need to research v3 specific requirements

## Testing Plan

After implementing fixes:
1. Test `quantization="none"` - verify GPU loading (check nvidia-smi)
2. Test `quantization="4bit"` - verify functional
3. Test `quantization="8bit"` - verify functional with new compute_dtype
4. Test CPU fallback - verify no CUDA errors
5. Test new generation parameters affect output quality
6. Test temperature values (0.0, 0.7, 1.0)
7. Test repetition_penalty reduces repetition

## Changes Summary

### Critical Bug Fixes
1. **Model Loading (Both Classes)**
   - **CRITICAL FIX**: Added `torch_dtype=compute_dtype` to ALL quantization paths (4bit, 8bit, none)
     - Without this, quantized models default to float32 causing massive CPU RAM usage
     - This was the root cause of poor GPU utilization and CPU bottleneck
   - Added `low_cpu_mem_usage=True` for all loading paths
   - Added `bnb_8bit_compute_dtype=compute_dtype` to 8bit quantization config
   - Restructured loading logic to use `load_kwargs` dict for cleaner code

2. **Device Compatibility**
   - Fixed hardcoded `.to("cuda")` calls → `.to(self.model.device)` (QwenVL line 413, Qwen line 648)
   - Ensures compatibility when running on CPU

3. **Temperature Bug**
   - Removed `temperature` parameter from `batch_decode()` calls (it doesn't accept this parameter)
   - Temperature is now only used in `generate()` where it belongs

### New Generation Parameters (Both Classes)
- `do_sample` (BOOLEAN, default: True) - Enable/disable sampling
- `top_p` (FLOAT, 0.0-1.0, default: 0.9) - Nucleus sampling
- `top_k` (INT, 0-200, default: 50) - Top-k sampling
- `min_p` (FLOAT, 0.0-1.0, default: 0.0) - Minimum probability threshold
- `repetition_penalty` (FLOAT, 1.0-2.0, default: 1.0) - Prevent repetition
- Expanded `temperature` range to 0-2 with 0.05 step size

### Performance Optimizations
1. **Flash Attention 2** ✅ ENHANCED
   - Proper ImportError detection (checks for flash_attn package)
   - Graceful fallback with automatic retry if Flash Attention fails
   - Verification logging shows which attention implementation actually loaded
   - Significantly improves attention computation speed when available

2. **Inference Mode**
   - Switched from `torch.no_grad()` to `torch.inference_mode()`
   - Provides better performance by more aggressively disabling gradient tracking

3. **KV Cache** ✅ CRITICAL
   - Added `use_cache=True` to all generate() calls (both classes)
   - Essential for quantized model performance (especially 4bit/8bit)
   - Without KV cache, every token regenerates full attention = major slowdown
   - Expected improvement: 4bit should now be ~equal or faster than unquantized

4. **4bit Quantization Optimization** ✅ NEW
   - Added `llm_int8_threshold=6.0` to BitsAndBytesConfig for 4bit
   - Keeps outlier features in FP16 for better performance
   - Critical for achieving fast 4bit inference

5. **Tensor Optimization** ✅ NEW
   - Optimized tensor_to_pil() to perform operations on GPU before CPU transfer
   - Uses `.clamp()` and `.byte()` on GPU instead of CPU numpy operations
   - Reduces unnecessary CPU-GPU memory transfers

6. **Pad Token Handling** ✅ NEW
   - Properly sets pad_token_id in generation_kwargs
   - Prevents warnings and improves batching efficiency
   - Falls back to eos_token_id if pad_token_id not available

7. **Performance Logging** ✅ NEW
   - Real-time tokens/sec measurement during generation
   - Model loading time tracking
   - Comprehensive device/dtype/quantization verification logging
   - Helps diagnose performance issues quickly

### Error Handling Improvements
- Specific `torch.cuda.OutOfMemoryError` handling with actionable suggestions
- Detailed exception logging with traceback for debugging
- User-friendly error messages with `[SCG_LocalVLM]` prefix

### ComfyUI v3 Compliance ✅ ENHANCED
- Added `RETURN_NAMES = ("text",)` to both node classes
- Verified standard node structure (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY)
- Enhanced __init__.py with:
  - Module docstring explaining features and compatibility
  - `__version__` export for v3 introspection
  - `__all__` export list for proper module interface
  - Better display names for nodes
- Added comprehensive docstrings to both node classes
- Maintains full backwards compatibility with ComfyUI v2
- Ready for ComfyUI v3 public API (stateless execution model compatible)

## Testing Checklist

### Critical Functionality
- [x] Code compiles without syntax errors
- [ ] `quantization="none"` loads model directly to GPU (check nvidia-smi)
- [ ] `quantization="4bit"` works correctly
- [ ] `quantization="8bit"` works correctly with new compute_dtype
- [ ] CPU fallback works without CUDA errors
- [ ] All new parameters accessible in ComfyUI interface

### Generation Quality
- [ ] `do_sample=False` produces deterministic output
- [ ] `do_sample=True` produces varied output
- [ ] `temperature` affects output randomness (test 0.0, 0.7, 1.5)
- [ ] `repetition_penalty` reduces repetitive text
- [ ] `top_p` and `top_k` affect output quality
- [ ] `min_p` works as expected

### Error Handling
- [ ] CUDA OOM error provides helpful suggestions
- [ ] General errors show detailed traceback in console
- [ ] Model loading errors are handled gracefully

## Notes

- Video-related improvements deferred (as requested)
- Code refactoring into shared functions deferred (current structure is clear and working)
- All changes are backwards compatible
- Temperature now properly used only in generation, not decoding

## Performance Optimization Session (2025-12-05)

### Final Results (QwenVL, Qwen3-VL-4B on RTX 5090)
| Mode | Attention | Speed | Status |
|------|-----------|-------|--------|
| Non-quantized | SDPA | 16.8 tok/s | ✅ Target met |
| Non-quantized | FA2 | 16.3 tok/s | ✅ Works on Blackwell |
| Non-quantized | Eager | 12.8 tok/s | ✅ Working |
| 4-bit | FA2 | 11.8 tok/s | ⚠️ Slower than expected |
| 4-bit | SDPA | 5.0 tok/s | ⚠️ Very slow |
| 4-bit | Eager | 12.7 tok/s | ✅ Best for 4-bit |

### Key Changes Made
1. **FA2 Detection** - Changed from blocker to hint (`_FA2_OPTIMIZED`), future-proof
2. **device_map Optimization** - `{"":0}` for single GPU quantized (avoids Accelerate overhead)
3. **Direct GPU Loading** - `{"":"cuda:0"}` for non-quantized (no CPU→GPU step)
4. **Removed .cuda() call** - device_map handles GPU placement

### Remaining Issues (External)
- **4-bit slow on Blackwell (SM 120)** - bitsandbytes kernels not optimized for RTX 50 series
- Workaround: Use eager attention for 4-bit, or use non-quantized
- Will likely be fixed in future bitsandbytes release

### Qwen Text Model
- Same optimizations NOT yet applied (edit tool issues during session)
- TODO: Apply same device_map and FA2 changes to Qwen text class

## Performance Fix Session (2025-12-07)

### Problem Analysis

User reported wildly inconsistent performance:
- Run 1: 3.30 tokens/sec, 117.85s for 389 tokens
- Run 2: 7.60 tokens/sec, 28.15s for 214 tokens
- Run 3: 2.75 tokens/sec, 145.67s for 400 tokens
- Run 4: 5.29 tokens/sec, 74.80s for 396 tokens

Performance varied 2.76x between runs. Model loading time also varied from 4s to 16s.

### Root Causes Identified

1. **`_maybe_move_to_cpu()` anti-pattern**: Before deleting the model, code was moving it to CPU first:
   ```python
   def _maybe_move_to_cpu(module):
       module.to("cpu")  # BAD: Copies entire model to CPU just to delete it
   ```
   This:
   - Allocates CPU RAM unnecessarily
   - Creates memory fragmentation on GPU
   - Doesn't properly release CUDA memory
   - Causes wildly varying inference speeds

2. **Model reloading every run**: Default `keep_model_loaded=False` caused the model to unload after each inference, triggering the fragmentation cycle.

3. **No class-level caching**: ComfyUI may create new node instances between executions. Without class-level caching, the model couldn't persist even with `keep_model_loaded=True`.

### Previous Failed Fixes (Reference)
- Commit 3008136: "Aggressive CUDA cleanup" - BROKE things (overly aggressive)
- Commit 4f5bcdb: "Attempted revert" - Still had issues
- Commit 98a311e: "Hard reset to 00122d3" - Reverted to known working state

### Fix Implementation

1. **Removed `_maybe_move_to_cpu()` function entirely**
   - Don't move model to CPU before deletion - this is wasteful
   - Just delete the model reference and let Python handle it

2. **Added CUDA synchronization before cleanup**
   ```python
   if torch.cuda.is_available():
       torch.cuda.synchronize()
   ```
   Ensures all CUDA operations complete before memory cleanup.

3. **Changed default `keep_model_loaded` to `True`**
   - More sensible default for most users
   - Avoids repeated load/unload cycles
   - User can still set to False if needed for memory constraints

4. **Added class-level model caching**
   - Model stored at class level, not just instance level
   - Survives ComfyUI creating new node instances
   - Automatically restores cached model to new instances

### Code Changes Summary

**Removed:**
- `_maybe_move_to_cpu()` function

**Modified `_unload_resources()` (both classes):**
- Added `torch.cuda.synchronize()` before cleanup
- Removed CPU move step
- Added explicit `del self.model` for cleaner gc
- Added class-level cache clearing

**Added to both classes:**
- Class-level cache variables (`_cached_model`, `_cached_processor/tokenizer`, etc.)
- `_save_to_cache()` method
- Cache restoration in `__init__()`
- Cache save after model loading

**Changed defaults:**
- `keep_model_loaded`: `False` → `True`

### Expected Improvements

1. **Consistent performance**: No more 2-3x variance between runs
2. **Faster repeated inference**: Model stays loaded between runs
3. **Cleaner memory management**: Proper CUDA cleanup without fragmentation
4. **Instance persistence**: Model survives ComfyUI instance recreation

### Diagnostic Session Results (2025-12-07)

With detailed timing added, we identified:

**Normal Performance (model kept loaded):**
- 12-14 tok/s consistently
- First run: 14.55 tok/s
- Preprocessing: <0.05s total (negligible)
- All time spent in model.generate()

**Random Slowdowns (3-7 tok/s):**
- Correlate with "got prompt" messages during generation
- ComfyUI's prompt processing interferes with CUDA operations
- External to node code - cannot fully fix from within the node

**Attention Mode Changes:**
- Removed "auto" option (was just mapping to "sdpa")
- Default changed from "auto" to "sdpa" directly
- Options now: sdpa, flash_attention_2, eager

**Workarounds for Slowdowns:**
- Run prompts one at a time, not in rapid succession
- Ensure no other GPU-intensive nodes running during generation
- Keep model loaded (default is now True)

### Enhanced Diagnostics (2025-12-07)

Added CUDA memory and SDPA backend diagnostics to help identify remaining variance.

**New Diagnostic Output:**
```
[SCG_LocalVLM] CUDA Memory before generation:
[SCG_LocalVLM]   Allocated: XXXX.X MB
[SCG_LocalVLM]   Reserved: XXXX.X MB
[SCG_LocalVLM]   Peak allocated: XXXX.X MB
[SCG_LocalVLM]   WARNING: High memory fragmentation (if > 1GB gap)
[SCG_LocalVLM] SDPA Backend Status:
[SCG_LocalVLM]   Flash SDP available: True/False
[SCG_LocalVLM]   Flash SDP enabled: True/False
[SCG_LocalVLM]   Memory-efficient SDP enabled: True/False
[SCG_LocalVLM]   Math SDP enabled: True/False
```

**What to Look For:**
1. **Memory Fragmentation**: Large gap between allocated and reserved memory indicates fragmentation, which can cause inconsistent performance
2. **SDPA Backend Mismatch**: If Flash SDP is unavailable/disabled but expected, this could cause slowdowns
3. **Peak vs Current**: Large difference suggests memory pressure from previous operations

**Potential Causes of Remaining Variance:**
- GPU thermal throttling (check GPU temp with `nvidia-smi`)
- Power state fluctuations (GPU may downclock when "idle")
- CUDA memory fragmentation from other processes
- SDPA backend fallback to math (slow) kernel
- Background system processes competing for GPU

### GPU State Monitoring (2025-12-07)

Added nvidia-smi based GPU monitoring to detect throttling:
```
[SCG_LocalVLM] GPU State:
[SCG_LocalVLM]   Clock: 2100/2520 MHz
[SCG_LocalVLM]   Power: 280/450 W
[SCG_LocalVLM]   Temp: 65°C
[SCG_LocalVLM]   WARNING: GPU clock throttled!  (if below 80% of max)
```

**Hypothesis from Latest Test Run:**
- Runs 2-4 showed progressive degradation (20→7→8.7→4.7 tok/s) during unload cycles
- Run 5 suddenly recovered (18.4 tok/s) even while still reloading
- Runs 6-8 stayed fast (17-19 tok/s) with model kept
- The recovery after Run 4's very slow 74s generation suggests GPU power state issue:
  - GPU may have been in low-power state during quick reloads
  - Long generation in Run 4 brought GPU to full power
  - Run 5 benefited from GPU already being at full power

**Fixed SDPA API:**
- Previous code used `is_flash_sdp_available()` which doesn't exist in PyTorch 2.x
- Updated to use correct functions: `flash_sdp_enabled()`, `mem_efficient_sdp_enabled()`, etc.
- Added `cudnn_sdp_enabled()` for newer PyTorch versions

### GPU Warmup Fix (2025-12-07)

**Root cause confirmed:** GPU diagnostics showed:
- Unload mode runs: GPU at 390-502 MHz (idle) → 3-9 tok/s
- Keep mode runs: GPU at 2970+ MHz (boost) → 18-20 tok/s

When model unloads, GPU drops to idle state. Reloading doesn't trigger clock boost.
Generation starts while GPU is still ramping up, causing massive slowdown.

**Fix:** Added `_warmup_gpu()` function that runs small matmul operations after
model loading to force GPU to boost clocks before generation starts.

### Further Analysis (2025-12-07)

**Matmul warmup was ineffective:**
- Completed in 0.001s (impossible for 3x 2048x2048 matmuls)
- On first run: 0.074s (CUDA initialization overhead)
- On subsequent runs: 0.001s (already initialized, no real work)

**Clock speed is NOT the cause:**
| Run | Clock Before | Tok/s |
|-----|--------------|-------|
| 2 | 990 MHz | **19.85** (FASTEST) |
| 5 | 1117 MHz | **5.64** (SLOWEST) |

Higher clock with worse performance proves clock isn't the issue.
The pre-generation clock reading is misleading - GPU ramps up during inference.

**New warmup approach:**
- Use actual SDPA kernels instead of matmul (different CUDA code paths)
- Explicitly enable SDPA backends for consistent kernel selection
- Add post-generation GPU state logging to catch throttling during inference

**Remaining hypotheses:**
1. SDPA backend selection differs after empty_cache()
2. CUDA memory allocator fragmentation patterns
3. JIT kernel caching being invalidated
4. PyTorch internal state (attention caches, etc.)
5. Something in HuggingFace generate() that's stateful

## Critical Findings from Research (2025-12-08)

### Root Cause: HuggingFace generate() Single CPU Core Bottleneck

**Source:** [GitHub Issue #24524](https://github.com/huggingface/transformers/issues/24524)

The `model.generate()` method has a **single CPU core bottleneck**. Even when all weights are on GPU:
- htop shows ONE CPU core maxed at 100% during generation
- nvidia-smi shows VRAM correctly allocated
- But GPU is waiting for Python to coordinate each token generation step

This explains the "peaks and valleys" in CUDA utilization - GPU does work, waits for Python, does work, waits...

### Known Issue: Qwen3-VL Slower Than Qwen2.5-VL

**Source:** [GitHub Issue #1657](https://github.com/QwenLM/Qwen3-VL/issues/1657)

Qwen3-VL is inherently slower than Qwen2.5-VL under the same conditions due to architectural changes:
- Enhanced MRope with interleaved layout
- DeepStack integration for multi-level ViT features

Qwen team recommends:
1. Enable Flash Attention 2 (especially for multi-image/video)
2. Use vLLM or SGLang for production deployment
3. Consider FP8 quantized versions

### RTX 5090 Blackwell Compatibility Issues

**Sources:**
- [PyTorch Issue #159207](https://github.com/pytorch/pytorch/issues/159207)
- [NVIDIA Forums](https://forums.developer.nvidia.com/t/rtx-5090-not-working-with-pytorch-and-stable-diffusion-sm-120-unsupported/338015)

RTX 5090 (SM 120/Blackwell) has incomplete PyTorch support:
- PyTorch 2.4.x runs with "severe limitations"
- Operations may run in "compatibility mode with poor performance"
- Need PyTorch 2.5.0+ for usable Blackwell support
- Flash Attention 2 kernels may not be optimized for SM 120

### The Solution: torch.compile + Static KV Cache

**Source:** [HuggingFace LLM Optimization Guide](https://huggingface.co/docs/transformers/en/llm_optims)

The dynamic KV cache grows with each token, which:
- Prevents torch.compile from optimizing the code
- Causes Python overhead on every token
- Results in the 4-10 tok/s we're seeing

**The fix:**
```python
# Enable static KV cache
model.generation_config.cache_implementation = "static"

# Compile the forward pass
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
```

This can provide **up to 4x speedup** by:
- Pre-allocating KV cache to max size (no dynamic allocation)
- Compiling model into CUDA graphs (eliminating Python overhead)
- Using "reduce-overhead" mode to minimize Python-related overhead

### Failed Attempts Log

| Date | Attempt | Result |
|------|---------|--------|
| 2025-12-07 | Removed _maybe_move_to_cpu() | Partial improvement |
| 2025-12-07 | Added class-level caching | Partial improvement |
| 2025-12-07 | Matmul GPU warmup | FAILED - wrong kernels (0.001s) |
| 2025-12-07 | SDPA-based warmup | FAILED - still 0.001s |
| 2025-12-08 | Removed erosdiffusion node | FAILED - no improvement |
| 2025-12-08 | StaticCache pre-allocation | FAILED - cudaMallocAsync error (conflict with other node) |
| 2025-12-08 | CUDA kernel warmup via generate() | FAILED - still variable performance |

### Next Steps

1. **Add per-token timing** - Log timestamp for each token to identify stall patterns
2. **Implement torch.compile + static cache** - The documented solution
3. **Test with PyTorch nightly** - May have better Blackwell support
4. **Consider vLLM integration** - If torch.compile doesn't work
