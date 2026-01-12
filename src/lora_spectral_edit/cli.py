"""
Command-line interface for LoRA spectral editing.
"""

import os
import gc
import json
import math
import random
import shutil
import argparse
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .io import (
    ensure_local_lora_dir,
    load_adapter_config,
    load_lora_state_dict,
    save_lora_state_dict,
    parse_lora_ab_key,
    layer_idx_from_module_prefix,
    get_scaling_for_module,
)
from .svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
from .hooks import ModuleSpec, HookContext, HOOK_CTX, register_sigma_hooks, remove_hooks
from .edit import EditConfig, apply_spectral_edit


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_calib_batch(tokenizer, examples: List[dict], add_eos: bool = True):
    """
    Build teacher-forcing inputs for calibration.

    Returns input_ids, attention_mask, labels tensors.
    Labels have -100 for prompt tokens (not included in loss).
    """
    input_ids_list = []
    labels_list = []

    for ex in examples:
        q = ex["question"]
        a = ex["answer"]
        prompt = f"Question: {q}\nAnswer:"
        full = prompt + " " + a
        if add_eos and tokenizer.eos_token:
            full = full + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(full, add_special_tokens=False).input_ids

        mask_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * mask_len + full_ids[mask_len:]

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attn_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)
    return input_ids, attn_mask, labels


def run_edit(args):
    """Main editing function."""
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This tool requires a CUDA GPU for gradient computation.")

    # Prepare LoRA files
    lora_dir = ensure_local_lora_dir(args.lora_path, cache_dir=args.cache_dir)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, fmt = load_lora_state_dict(lora_dir)

    # Copy adapter directory to out_dir
    if os.path.abspath(args.out_dir) != os.path.abspath(lora_dir):
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
        shutil.copytree(lora_dir, args.out_dir)

    # Identify LoRA A/B pairs for selected modules/layers
    pairs: Dict[str, dict] = {}
    target_modules_set = set(args.target_modules)

    for k, t in sd.items():
        parsed = parse_lora_ab_key(k)
        if not parsed:
            continue
        prefix, which, adapter = parsed

        suffix = prefix.split(".")[-1]
        if suffix not in target_modules_set:
            continue

        li = layer_idx_from_module_prefix(prefix)
        if li is not None and not (args.layer_min <= li <= args.layer_max):
            continue

        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (k, t, adapter)

    selected_prefixes = [p for p in pairs.keys() if "A" in pairs[p] and "B" in pairs[p]]
    if not selected_prefixes:
        raise RuntimeError("No matching LoRA (A,B) pairs found for given target_modules/layer range.")

    print(f"[Info] Selected LoRA modules: {len(selected_prefixes)}")
    for p in selected_prefixes[:5]:
        print(f"   - {p}")
    if len(selected_prefixes) > 5:
        print(f"   ... and {len(selected_prefixes) - 5} more")

    # Load model for gradient probing
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)

    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=True).to(device)
    model.eval()
    model.config.use_cache = False

    # Freeze everything except LoRA params
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    # Build ModuleSpec for each selected module
    name_to_module = dict(model.named_modules())
    specs: Dict[str, ModuleSpec] = {}

    for prefix in selected_prefixes:
        keyA, A_cpu, adapterA = pairs[prefix]["A"]
        keyB, B_cpu, adapterB = pairs[prefix]["B"]
        adapter_name = adapterA if adapterA is not None else adapterB

        if prefix not in name_to_module:
            candidates = [nm for nm in name_to_module.keys() if nm.endswith(prefix)]
            if not candidates:
                raise RuntimeError(f"Cannot find module '{prefix}' in model")
            module_name = candidates[0]
        else:
            module_name = prefix

        mod = name_to_module[module_name]
        A = A_cpu.to(device)
        B = B_cpu.to(device)

        U, S, Vh, V = lowrank_svd_from_ba(B, A)
        scaling = get_scaling_for_module(adapter_cfg, prefix)

        specs[prefix] = ModuleSpec(
            module_prefix=prefix,
            module=mod,
            U=U.detach(),
            V=V.detach(),
            Vh=Vh.detach(),
            sigma0=S.detach().cpu(),
            scaling=scaling,
            adapter=adapter_name,
        )

    print(f"[Info] Built SVD specs for {len(specs)} modules.")

    # Register hooks and run calibration
    handles = register_sigma_hooks(specs)

    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"]

    ncal = min(args.calib_samples, len(train_ds))
    calib_examples = [train_ds[i] for i in range(ncal)]

    bs = max(1, args.calib_batch_size)
    total_loss = 0.0
    n_steps = 0

    HOOK_CTX.reset()

    for i in range(0, ncal, bs):
        batch_ex = calib_examples[i:i+bs]
        input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, add_eos=True)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        HOOK_CTX.attn_mask = attn_mask

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        total_loss += float(loss.item())
        n_steps += 1

        model.zero_grad(set_to_none=True)
        loss.backward()
        model.zero_grad(set_to_none=True)

        step_num = i // bs + 1
        total_steps = math.ceil(ncal / bs)
        if step_num % 5 == 0 or step_num == total_steps:
            print(f"[Calib] step {step_num}/{total_steps} loss={loss.item():.4f}")

    remove_hooks(handles)
    HOOK_CTX.attn_mask = None

    print(f"[Calib] avg loss: {total_loss / max(1, n_steps):.4f}")
    print(f"[Calib] total active tokens: {HOOK_CTX.total_active_tokens}")

    if not HOOK_CTX.gsum:
        raise RuntimeError("No gradients accumulated. Hooks may not have fired.")

    # Build edit config
    edit_config = EditConfig(
        mode=args.mode,
        core_frac=args.core_frac,
        noise_frac=args.noise_frac,
        amp_factor=args.amp_factor,
        sup_factor=args.sup_factor,
        mid_factor=args.mid_factor,
        min_core_k=args.min_core_k,
        smooth_temperature=args.smooth_temperature,
        smooth_center_q=args.smooth_center_q,
        smooth_align_mid=not args.no_smooth_align_mid,
        z_high=args.z_high,
        z_low=args.z_low,
        z_tau=args.z_tau,
        z_fallback_std=args.z_fallback_std,
        robust_z_high=args.robust_z_high,
        robust_z_low=args.robust_z_low,
        robust_z_tau=args.robust_z_tau,
        robust_fallback_sigma=args.robust_fallback_sigma,
        eta=args.eta,
        update_mode=args.update_mode,
        asymmetric_update=args.asymmetric_update,
        eta_suppress=args.eta_suppress,
        eta_enhance=args.eta_enhance,
        pos_power=args.pos_power,
        grad_norm=args.grad_norm,
        preserve_energy=args.preserve_energy,
        sigma_clip_min=args.sigma_clip_min,
    )

    # Apply edits
    sigma_stats = {}

    for prefix, spec in specs.items():
        sigma0 = spec.sigma0.clone()
        g = HOOK_CTX.gsum.get(prefix, None)
        if g is None:
            continue

        sigma_new, stats = apply_spectral_edit(sigma0, g, edit_config)

        # Rebuild A/B
        U = spec.U.to(device)
        Vh = spec.Vh.to(device)
        sigma_new_gpu = sigma_new.to(device)

        B_new, A_new = rebuild_ba_from_uv_sigma(U, Vh, sigma_new_gpu)

        keyA, A_old, _ = pairs[prefix]["A"]
        keyB, B_old, _ = pairs[prefix]["B"]
        A_new = A_new.to(dtype=A_old.dtype).detach().cpu()
        B_new = B_new.to(dtype=B_old.dtype).detach().cpu()

        sd[keyA] = A_new
        sd[keyB] = B_new
        sigma_stats[prefix] = stats

    # Save edited adapter
    save_lora_state_dict(args.out_dir, sd, fmt)

    meta = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "target_modules": args.target_modules,
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "calib_samples": args.calib_samples,
        "mode": args.mode,
        "core_frac": args.core_frac,
        "noise_frac": args.noise_frac,
        "amp_factor": args.amp_factor,
        "sup_factor": args.sup_factor,
        "mid_factor": args.mid_factor,
        "grad_norm": args.grad_norm,
        "preserve_energy": args.preserve_energy,
        "seed": args.seed,
    }

    with open(os.path.join(args.out_dir, "spectral_edit_meta.json"), "w") as f:
        json.dump({"meta": meta, "sigma_stats": sigma_stats}, f, indent=2)

    print(f"[Save] Edited adapter saved to: {args.out_dir}")

    # Free GPU memory
    try:
        model.to("cpu")
        base.to("cpu")
    except Exception:
        pass
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    # Optional evaluation
    if args.eval_gsm8k:
        try:
            from .eval_gsm8k import evaluate_both_loras
        except ImportError:
            print("[Warn] vLLM not installed; skipping evaluation.")
            return

        print("\n[Eval] Running GSM8K evaluation...")
        results = evaluate_both_loras(
            base_model_id=args.base_model,
            baseline_lora_dir=lora_dir,
            edited_lora_dir=args.out_dir,
            fewshot_k=args.eval_fewshot,
            max_samples=args.eval_max_samples,
            temperature=args.eval_temperature,
            max_tokens=args.eval_max_tokens,
            seed=args.seed,
            out_dir=args.out_dir,
        )
        print(f"[Baseline] {results['baseline']}")
        print(f"[Edited]   {results['edited']}")

        with open(os.path.join(args.out_dir, "metrics_gsm8k.json"), "w") as f:
            json.dump({"results": results, "meta": meta}, f, indent=2)


def run_eval(args):
    """Standalone evaluation function with profile support."""
    # Determine which profile to use
    profile_name = getattr(args, 'eval_profile', None)

    if profile_name:
        # Use profile-based evaluation
        from .eval_profiles import get_profile
        profile = get_profile(profile_name)

        if profile.task == "gsm8k":
            from .eval_gsm8k import evaluate_gsm8k_with_profile

            print(f"[Eval] Using profile '{profile_name}': {profile.description}")
            result = evaluate_gsm8k_with_profile(
                base_model_id=args.base_model,
                lora_dir=args.lora_dir,
                profile_name=profile_name,
                max_samples=args.max_samples if args.max_samples > 0 else None,
                seed=args.seed,
                max_model_len=getattr(args, 'max_model_len', 4096),
                out_dir=args.out_dir if hasattr(args, 'out_dir') and args.out_dir else args.lora_dir,
            )
            print(f"[Result] acc={result['acc']:.4f} ({result['correct']}/{result['total']})")
            print(f"[Config] {result.get('config_path', 'N/A')}")

        elif profile.task == "humaneval":
            from .eval_humaneval import evaluate_humaneval_with_profile

            print(f"[Eval] Using profile '{profile_name}': {profile.description}")
            out_dir = args.out_dir if hasattr(args, 'out_dir') and args.out_dir else args.lora_dir
            result = evaluate_humaneval_with_profile(
                base_model_id=args.base_model,
                lora_dir=args.lora_dir,
                profile_name=profile_name,
                max_samples=args.max_samples if args.max_samples > 0 else None,
                seed=args.seed,
                max_model_len=getattr(args, 'max_model_len', 4096),
                out_dir=out_dir,
                config_src=getattr(args, 'config_src', None),
            )
            print(f"[Result] pass@{profile.pass_k}={result.get(f'pass@{profile.pass_k}', 'N/A')}")
            print(f"[Config] {result.get('config_path', 'N/A')}")

        else:
            raise ValueError(f"Unknown task in profile: {profile.task}")

    else:
        # Legacy GSM8K evaluation
        from .eval_gsm8k import evaluate_gsm8k_vllm

        print(f"[Eval] Evaluating {args.lora_dir} on GSM8K...")
        out_dir = args.out_dir if hasattr(args, 'out_dir') and args.out_dir else args.lora_dir
        result = evaluate_gsm8k_vllm(
            base_model_id=args.base_model,
            lora_dir=args.lora_dir,
            fewshot_k=args.fewshot,
            max_samples=args.max_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            out_dir=out_dir,
        )
        print(f"[Result] {result}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Save] Metrics saved to: {args.out_json}")


def run_eval_humaneval(args):
    """Standalone HumanEval evaluation function."""
    from .eval_profiles import get_profile

    profile_name = args.eval_profile if args.eval_profile else "greedy_code"

    if args.eval_profile:
        from .eval_humaneval import evaluate_humaneval_with_profile

        profile = get_profile(profile_name)
        print(f"[Eval] Using profile '{profile_name}': {profile.description}")

        out_dir = args.out_dir if args.out_dir else args.lora_dir
        result = evaluate_humaneval_with_profile(
            base_model_id=args.base_model,
            lora_dir=args.lora_dir if not args.base_only else None,
            profile_name=profile_name,
            max_samples=args.max_samples if args.max_samples > 0 else None,
            seed=args.seed,
            max_model_len=args.max_model_len,
            out_dir=out_dir,
            config_src=args.config_src,
            eval_n_workers=args.eval_n_workers,
            eval_timeout=args.eval_timeout,
        )

        print(f"[Result] pass@{profile.pass_k}={result.get(f'pass@{profile.pass_k}', 'N/A')}")
        print(f"[Config] {result.get('config_path', 'N/A')}")

    else:
        # Manual params mode
        from .eval_humaneval import (
            HAVE_VLLM, HAVE_HUMAN_EVAL, LLM, SamplingParams,
            read_problems, load_adapter_config, ensure_lora_has_config,
            infer_max_lora_rank, eval_one_adapter_multisample, eval_one_adapter,
            write_eval_config
        )

        if not HAVE_VLLM:
            raise RuntimeError("vLLM not available. Please `pip install vllm`.")
        if not HAVE_HUMAN_EVAL:
            raise RuntimeError("human_eval not available.")

        out_dir = args.out_dir if args.out_dir else (args.lora_dir if args.lora_dir else ".")
        os.makedirs(out_dir, exist_ok=True)

        # Setup
        enable_lora = not args.base_only and args.lora_dir is not None
        max_r = 16

        if args.lora_dir:
            ensure_lora_has_config(args.lora_dir, args.config_src)
            cfg = load_adapter_config(args.lora_dir)
            max_r = max(max_r, infer_max_lora_rank(cfg))

        # Load problems
        print("[Data] Loading HumanEval problems...")
        problems = read_problems()
        task_ids = sorted(problems.keys())

        if args.max_samples > 0:
            task_ids = task_ids[:args.max_samples]

        prompts = [problems[tid]["prompt"] for tid in task_ids]
        print(f"[Data] Loaded {len(prompts)} HumanEval tasks.")

        # Init vLLM
        print(f"[vLLM] Initializing LLM engine (Base: {args.base_model})...")
        llm = LLM(
            model=args.base_model,
            dtype="float16",
            max_model_len=args.max_model_len,
            enable_lora=enable_lora,
            max_lora_rank=max_r if enable_lora else 16,
            seed=args.seed,
        )

        # Evaluate
        if args.n_generations > 1:
            result = eval_one_adapter_multisample(
                llm=llm,
                prompts=prompts,
                task_ids=task_ids,
                n_generations=args.n_generations,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                stop_words=args.stop_words,
                lora_dir=args.lora_dir if not args.base_only else None,
                adapter_name="lora" if args.lora_dir else "base",
                out_dir=out_dir,
                eval_n_workers=args.eval_n_workers,
                eval_timeout=args.eval_timeout,
                pass_k=args.pass_k,
            )
            print(f"[Result] pass@{args.pass_k}={result.get(f'pass@{args.pass_k}', 'N/A')}")
        else:
            sp = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                stop=args.stop_words,
            )
            result = eval_one_adapter(
                llm=llm,
                sp=sp,
                prompts=prompts,
                task_ids=task_ids,
                lora_dir=args.lora_dir if not args.base_only else None,
                adapter_name="lora" if args.lora_dir else "base",
                out_dir=out_dir,
                eval_n_workers=args.eval_n_workers,
                eval_timeout=args.eval_timeout,
            )
            print(f"[Result] pass@1={result.get('pass@1', 'N/A')}")

        # Write config
        config_path = write_eval_config(
            out_dir=out_dir,
            task="humaneval",
            n_generations=args.n_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            total_tasks=len(task_ids),
            metric_name=f"pass@{args.pass_k}" if args.n_generations > 1 else "pass@1",
            score=result.get(f"pass@{args.pass_k}", result.get("pass@1", 0.0)),
            extra_meta={
                "base_model_id": args.base_model,
                "lora_dir": args.lora_dir,
                "seed": args.seed,
            },
        )
        result["config_path"] = config_path
        print(f"[Config] {config_path}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Save] Metrics saved to: {args.out_json}")


def run_smooth_abs_experiment(args):
    """Run smooth_abs experiment: edit + evaluate on GSM8K and HumanEval."""
    # Import the experiment runner
    import sys
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exp_path = os.path.join(ROOT_DIR, "experiments")
    if exp_path not in sys.path:
        sys.path.insert(0, exp_path)

    from run_smooth_abs_experiment import (
        run_smooth_abs_edit,
        run_gsm8k_eval,
        run_humaneval_eval,
        write_summary,
    )

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Define paths
    edited_adapter_dir = os.path.join(args.out_dir, "adapter_smooth_abs")
    gsm8k_edited_dir = os.path.join(args.out_dir, "gsm8k_edited")
    gsm8k_baseline_dir = os.path.join(args.out_dir, "gsm8k_baseline")
    humaneval_edited_dir = os.path.join(args.out_dir, "humaneval_edited")
    humaneval_baseline_dir = os.path.join(args.out_dir, "humaneval_baseline")

    smooth_align_mid = not args.no_smooth_align_mid

    print("=" * 70)
    print("smooth_abs Experiment")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"LoRA: {args.lora_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Hyperparams: amp={args.amp_factor}, sup={args.sup_factor}, temp={args.smooth_temperature}")
    print(f"Do baseline eval: {args.do_baseline_eval}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print("=" * 70)

    # Step 1: Run smooth_abs edit
    edit_meta = {}
    if not args.skip_edit:
        print("\n" + "=" * 70)
        print("Phase 1: smooth_abs Spectral Editing")
        print("=" * 70)

        edit_meta = run_smooth_abs_edit(
            base_model_id=args.base_model,
            lora_dir=args.lora_dir,
            out_adapter_dir=edited_adapter_dir,
            target_modules=args.target_modules,
            layer_min=0,
            layer_max=10**9,
            calib_samples=args.calib_samples,
            calib_batch_size=args.calib_batch_size,
            amp_factor=args.amp_factor,
            sup_factor=args.sup_factor,
            mid_factor=args.mid_factor,
            smooth_temperature=args.smooth_temperature,
            smooth_center_q=args.smooth_center_q,
            smooth_align_mid=smooth_align_mid,
            grad_norm=args.grad_norm,
            preserve_energy=args.preserve_energy,
            sigma_clip_min=0.0,
            seed=args.seed,
            cache_dir=None,
        )

        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[Skip] Skipping edit phase (--skip_edit)")
        meta_path = os.path.join(edited_adapter_dir, "spectral_edit_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                edit_meta = json.load(f)

    # Step 2: GSM8K evaluation
    baseline_gsm8k = None
    edited_gsm8k = None

    if not args.skip_gsm8k:
        print("\n" + "=" * 70)
        print("Phase 2: GSM8K Evaluation (paper_math profile)")
        print("=" * 70)

        if args.do_baseline_eval:
            print("\n[GSM8K] Evaluating baseline...")
            baseline_gsm8k = run_gsm8k_eval(
                base_model_id=args.base_model,
                lora_dir=args.lora_dir,
                out_dir=gsm8k_baseline_dir,
                max_samples=args.max_samples,
                seed=args.seed,
                max_model_len=args.max_model_len,
            )
            gc.collect()
            torch.cuda.empty_cache()

        print("\n[GSM8K] Evaluating edited adapter...")
        edited_gsm8k = run_gsm8k_eval(
            base_model_id=args.base_model,
            lora_dir=edited_adapter_dir,
            out_dir=gsm8k_edited_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            max_model_len=args.max_model_len,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # Step 3: HumanEval evaluation
    baseline_humaneval = None
    edited_humaneval = None

    if not args.skip_humaneval:
        print("\n" + "=" * 70)
        print("Phase 3: HumanEval Evaluation (paper_code_main settings)")
        print("=" * 70)

        if args.do_baseline_eval:
            print("\n[HumanEval] Evaluating baseline...")
            baseline_humaneval = run_humaneval_eval(
                base_model_id=args.base_model,
                lora_dir=args.lora_dir,
                out_dir=humaneval_baseline_dir,
                max_samples=args.max_samples,
                n_generations=args.humaneval_n_generations,
                seed=args.seed,
                max_model_len=args.max_model_len,
            )
            gc.collect()
            torch.cuda.empty_cache()

        print("\n[HumanEval] Evaluating edited adapter...")
        edited_humaneval = run_humaneval_eval(
            base_model_id=args.base_model,
            lora_dir=edited_adapter_dir,
            out_dir=humaneval_edited_dir,
            max_samples=args.max_samples,
            n_generations=args.humaneval_n_generations,
            seed=args.seed,
            max_model_len=args.max_model_len,
            config_src=args.lora_dir,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # Step 4: Write summary
    print("\n" + "=" * 70)
    print("Phase 4: Writing Summary")
    print("=" * 70)

    if edited_gsm8k or edited_humaneval:
        write_summary(
            out_dir=args.out_dir,
            edit_meta=edit_meta,
            baseline_gsm8k=baseline_gsm8k,
            edited_gsm8k=edited_gsm8k or {},
            baseline_humaneval=baseline_humaneval,
            edited_humaneval=edited_humaneval or {},
        )

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {args.out_dir}")
    print(f"  - Edited adapter: adapter_smooth_abs/")
    print(f"  - Summary: summary.json, summary.csv")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Spectral Edit - Sensitivity-based spectral editing for LoRA adapters"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Edit subcommand
    edit_parser = subparsers.add_parser("edit", help="Edit a LoRA adapter using spectral manipulation")
    edit_parser.add_argument("--base_model", type=str, required=True,
                             help="HuggingFace model ID for base model")
    edit_parser.add_argument("--lora_path", type=str, required=True,
                             help="Path or HF ID for LoRA adapter")
    edit_parser.add_argument("--out_dir", type=str, required=True,
                             help="Output directory for edited adapter")

    edit_parser.add_argument("--target_modules", type=str, nargs="+", default=["down_proj", "o_proj"],
                             help="Module names to edit (default: down_proj o_proj)")
    edit_parser.add_argument("--layer_min", type=int, default=0,
                             help="Minimum layer index to edit")
    edit_parser.add_argument("--layer_max", type=int, default=10**9,
                             help="Maximum layer index to edit")

    edit_parser.add_argument("--calib_samples", type=int, default=32,
                             help="Number of calibration samples")
    edit_parser.add_argument("--calib_batch_size", type=int, default=2,
                             help="Calibration batch size")

    edit_parser.add_argument("--mode", "--edit_mode", type=str,
                             choices=["abs_select", "smooth_abs", "double_smooth", "z_score",
                                      "robust_z", "random_index", "gd"],
                             default="abs_select",
                             help="Edit mode: abs_select, smooth_abs, double_smooth, z_score, robust_z, random_index, or gd")
    edit_parser.add_argument("--core_frac", type=float, default=0.2,
                             help="Fraction of dims to amplify (abs_select mode)")
    edit_parser.add_argument("--noise_frac", type=float, default=0.2,
                             help="Fraction of dims to suppress (abs_select mode)")
    edit_parser.add_argument("--amp_factor", type=float, default=1.25,
                             help="Amplification factor for core dims")
    edit_parser.add_argument("--sup_factor", type=float, default=0.80,
                             help="Suppression factor for noise dims")
    edit_parser.add_argument("--mid_factor", type=float, default=1.0,
                             help="Scale factor for middle dims")
    edit_parser.add_argument("--min_core_k", type=int, default=1,
                             help="Minimum number of core dims per module")

    edit_parser.add_argument("--smooth_temperature", type=float, default=0.35,
                             help="Smoothness for smooth_abs/double_smooth (larger=smoother, smaller=sharper)")
    edit_parser.add_argument("--smooth_center_q", type=float, default=0.5,
                             help="Center quantile for smooth_abs (0.5=median)")
    edit_parser.add_argument("--no_smooth_align_mid", action="store_true",
                             help="Disable aligning gate(center)=mid_factor in smooth_abs")

    edit_parser.add_argument("--z_high", type=float, default=1.0,
                             help="Z-score threshold for amplification (z_score mode)")
    edit_parser.add_argument("--z_low", type=float, default=-0.5,
                             help="Z-score threshold for suppression (z_score mode)")
    edit_parser.add_argument("--z_tau", type=float, default=0.2,
                             help="Temperature for z-score gating (z_score mode)")
    edit_parser.add_argument("--z_fallback_std", type=float, default=1e-6,
                             help="Stddev floor that triggers z_score fallback")

    edit_parser.add_argument("--robust_z_high", type=float, default=1.0,
                             help="Robust z-score threshold for amplification (robust_z mode)")
    edit_parser.add_argument("--robust_z_low", type=float, default=-0.5,
                             help="Robust z-score threshold for suppression (robust_z mode)")
    edit_parser.add_argument("--robust_z_tau", type=float, default=0.2,
                             help="Temperature for robust z-score gating (robust_z mode)")
    edit_parser.add_argument("--robust_fallback_sigma", type=float, default=1e-6,
                             help="Sigma floor that triggers robust_z fallback")

    edit_parser.add_argument("--eta", type=float, default=0.2,
                             help="Learning rate (gd mode)")
    edit_parser.add_argument("--update_mode", type=str, choices=["additive", "multiplicative"],
                             default="multiplicative", help="Update mode (gd mode)")
    edit_parser.add_argument("--asymmetric_update", action="store_true",
                             help="Use asymmetric step sizes (gd mode)")
    edit_parser.add_argument("--eta_suppress", type=float, default=2.0,
                             help="Step size for g>0 (gd mode)")
    edit_parser.add_argument("--eta_enhance", type=float, default=0.2,
                             help="Step size for g<0 (gd mode)")
    edit_parser.add_argument("--pos_power", type=float, default=1.0,
                             help="Nonlinearity power (gd mode)")

    edit_parser.add_argument("--grad_norm", type=str, choices=["none", "mean_abs", "l2"],
                             default="mean_abs", help="Gradient normalization method")
    edit_parser.add_argument("--preserve_energy", type=str, choices=["none", "l1", "l2"],
                             default="l1", help="Energy preservation method")
    edit_parser.add_argument("--sigma_clip_min", type=float, default=0.0,
                             help="Minimum sigma value after editing")

    edit_parser.add_argument("--eval_gsm8k", action="store_true",
                             help="Run GSM8K evaluation after editing (requires vLLM)")
    edit_parser.add_argument("--eval_fewshot", type=int, default=5,
                             help="Number of few-shot examples for evaluation")
    edit_parser.add_argument("--eval_max_samples", type=int, default=-1,
                             help="Max eval samples (-1 for all)")
    edit_parser.add_argument("--eval_temperature", type=float, default=0.0,
                             help="Sampling temperature for evaluation")
    edit_parser.add_argument("--eval_max_tokens", type=int, default=512,
                             help="Max tokens to generate for evaluation")

    edit_parser.add_argument("--cache_dir", type=str, default=None,
                             help="Cache directory for HF downloads")
    edit_parser.add_argument("--seed", type=int, default=0,
                             help="Random seed")

    # Eval subcommand (GSM8K or profile-based)
    eval_parser = subparsers.add_parser("eval", help="Evaluate a LoRA adapter (requires vLLM)")
    eval_parser.add_argument("--base_model", type=str, required=True,
                             help="HuggingFace model ID for base model")
    eval_parser.add_argument("--lora_dir", type=str, required=True,
                             help="Path to LoRA adapter directory")

    # Eval profile support
    eval_parser.add_argument("--eval_profile", type=str, default=None,
                             choices=["paper_math", "paper_code_main", "greedy_math", "greedy_code"],
                             help="Use predefined evaluation profile. "
                                  "paper_math: GSM8K 5-shot greedy strict match. "
                                  "paper_code_main: HumanEval 50-sample pass@1 (temp=0.2, top_p=0.95). "
                                  "greedy_math: GSM8K greedy (legacy). "
                                  "greedy_code: HumanEval greedy (legacy).")

    eval_parser.add_argument("--fewshot", type=int, default=5,
                             help="Number of few-shot examples (GSM8K)")
    eval_parser.add_argument("--max_samples", type=int, default=-1,
                             help="Max test samples (-1 for all)")
    eval_parser.add_argument("--temperature", type=float, default=0.0,
                             help="Sampling temperature")
    eval_parser.add_argument("--max_tokens", type=int, default=512,
                             help="Max tokens to generate")
    eval_parser.add_argument("--max_model_len", type=int, default=4096,
                             help="Max model context length for vLLM")
    eval_parser.add_argument("--seed", type=int, default=0,
                             help="Random seed")
    eval_parser.add_argument("--out_json", type=str, default=None,
                             help="Output JSON file for metrics")
    eval_parser.add_argument("--out_dir", type=str, default=None,
                             help="Output directory for results (defaults to lora_dir)")
    eval_parser.add_argument("--config_src", type=str, default=None,
                             help="Source for adapter_config.json if missing (HumanEval)")

    # HumanEval subcommand
    humaneval_parser = subparsers.add_parser("eval-humaneval",
                                              help="Evaluate on HumanEval (requires vLLM + human_eval)")
    humaneval_parser.add_argument("--base_model", type=str, required=True,
                                   help="HuggingFace model ID for base model")
    humaneval_parser.add_argument("--lora_dir", type=str, default=None,
                                   help="Path to LoRA adapter directory")
    humaneval_parser.add_argument("--base_only", action="store_true",
                                   help="Evaluate base model only (no LoRA)")
    humaneval_parser.add_argument("--config_src", type=str, default=None,
                                   help="Source for adapter_config.json if missing")

    humaneval_parser.add_argument("--eval_profile", type=str, default=None,
                                   choices=["paper_code_main", "greedy_code"],
                                   help="Use predefined evaluation profile. "
                                        "paper_code_main: 50 samples, temp=0.2, top_p=0.95, pass@1. "
                                        "greedy_code: 1 sample, greedy decoding.")

    humaneval_parser.add_argument("--max_samples", type=int, default=-1,
                                   help="Max HumanEval tasks (-1 for all 164)")
    humaneval_parser.add_argument("--n_generations", type=int, default=1,
                                   help="Generations per task (paper_code_main: 50)")
    humaneval_parser.add_argument("--temperature", type=float, default=0.0,
                                   help="Sampling temperature (paper_code_main: 0.2)")
    humaneval_parser.add_argument("--top_p", type=float, default=1.0,
                                   help="Top-p sampling (paper_code_main: 0.95)")
    humaneval_parser.add_argument("--max_tokens", type=int, default=512,
                                   help="Max tokens to generate")
    humaneval_parser.add_argument("--pass_k", type=int, default=1,
                                   help="k for pass@k metric")
    humaneval_parser.add_argument("--max_model_len", type=int, default=4096,
                                   help="Max model context length for vLLM")
    humaneval_parser.add_argument("--seed", type=int, default=0,
                                   help="Random seed")
    humaneval_parser.add_argument("--out_json", type=str, default=None,
                                   help="Output JSON file for metrics")
    humaneval_parser.add_argument("--out_dir", type=str, default=None,
                                   help="Output directory for results")
    humaneval_parser.add_argument("--eval_n_workers", type=int, default=4,
                                   help="Workers for code execution")
    humaneval_parser.add_argument("--eval_timeout", type=float, default=3.0,
                                   help="Timeout per test case")
    humaneval_parser.add_argument("--stop_words", type=str, nargs="*",
                                   default=["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint"],
                                   help="Stop sequences for code generation")

    # run-smooth-abs subcommand (end-to-end experiment)
    smoothabs_parser = subparsers.add_parser(
        "run-smooth-abs",
        help="Run smooth_abs experiment: edit + evaluate on GSM8K and HumanEval"
    )
    smoothabs_parser.add_argument("--base_model", type=str, required=True,
                                   help="HuggingFace model ID for base model")
    smoothabs_parser.add_argument("--lora_dir", type=str, required=True,
                                   help="Path or HF ID for LoRA adapter")
    smoothabs_parser.add_argument("--out_dir", type=str, required=True,
                                   help="Output directory for experiment results")

    # smooth_abs hyperparams
    smoothabs_parser.add_argument("--amp_factor", type=float, default=1.25,
                                   help="Amplification factor for high-sensitivity dims")
    smoothabs_parser.add_argument("--sup_factor", type=float, default=0.8,
                                   help="Suppression factor for low-sensitivity dims")
    smoothabs_parser.add_argument("--mid_factor", type=float, default=1.0,
                                   help="Scale factor for middle dims")
    smoothabs_parser.add_argument("--smooth_temperature", type=float, default=0.35,
                                   help="Temperature for sigmoid gate (larger=smoother)")
    smoothabs_parser.add_argument("--smooth_center_q", type=float, default=0.5,
                                   help="Center quantile for sigmoid (0.5=median)")
    smoothabs_parser.add_argument("--no_smooth_align_mid", action="store_true",
                                   help="Disable aligning gate(center)=mid_factor")

    # Edit params
    smoothabs_parser.add_argument("--target_modules", type=str, nargs="+",
                                   default=["down_proj", "o_proj"],
                                   help="Module names to edit")
    smoothabs_parser.add_argument("--calib_samples", type=int, default=256,
                                   help="Number of calibration samples")
    smoothabs_parser.add_argument("--calib_batch_size", type=int, default=2,
                                   help="Calibration batch size")
    smoothabs_parser.add_argument("--grad_norm", type=str, choices=["none", "mean_abs", "l2"],
                                   default="mean_abs", help="Gradient normalization")
    smoothabs_parser.add_argument("--preserve_energy", type=str, choices=["none", "l1", "l2"],
                                   default="l1", help="Energy preservation method")

    # Eval params
    smoothabs_parser.add_argument("--do_baseline_eval", action="store_true",
                                   help="Also evaluate the baseline (unedited) adapter")
    smoothabs_parser.add_argument("--max_samples", type=int, default=None,
                                   help="Max samples for both GSM8K and HumanEval (for smoke tests)")
    smoothabs_parser.add_argument("--humaneval_n_generations", type=int, default=50,
                                   help="Generations per HumanEval problem (paper: 50)")
    smoothabs_parser.add_argument("--max_model_len", type=int, default=4096,
                                   help="Max model context length for vLLM")
    smoothabs_parser.add_argument("--seed", type=int, default=42,
                                   help="Random seed")

    # Skip flags
    smoothabs_parser.add_argument("--skip_edit", action="store_true",
                                   help="Skip editing, only run eval")
    smoothabs_parser.add_argument("--skip_gsm8k", action="store_true",
                                   help="Skip GSM8K evaluation")
    smoothabs_parser.add_argument("--skip_humaneval", action="store_true",
                                   help="Skip HumanEval evaluation")

    args = parser.parse_args()

    if args.command == "edit":
        run_edit(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "eval-humaneval":
        run_eval_humaneval(args)
    elif args.command == "run-smooth-abs":
        run_smooth_abs_experiment(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
