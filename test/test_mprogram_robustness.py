"""
Robustness test for a Mellea m-program using BenchDrift variations.

Generates semantic variations of a problem, tests each through the m-program,
and reports pass/fail with color-coded output.

    python test_mprogram_robustness.py --top-k 5 --no-enrich
    python test_mprogram_robustness.py --backend-model mistral:7b --top-k 3
    python test_mprogram_robustness.py --gen-model qwen3:8b --use-axes all
"""
import sys
import os
import re
import io
import contextlib
import argparse
import logging
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
from pathlib import Path

from mellea import start_session
from mellea.backends.types import ModelOption
from mellea_contribs.tools.benchdrift_runner import (
    run_benchdrift_pipeline,
    analyze_robustness,
)

# ANSI
G = "\033[92m"   # green
R = "\033[91m"   # red
Y = "\033[93m"   # yellow
D = "\033[2m"    # dim
B = "\033[1m"    # bold
X = "\033[0m"    # reset


def _suppress_noise():
    for name in ['BenchDrift', 'benchdrift', 'mellea_contribs', 'mellea',
                 'httpx', 'httpcore', 'urllib3', 'requests']:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    try:
        from mellea.helpers.fancy_logger import FancyLogger
        fl = FancyLogger.get_logger()
        fl.setLevel(logging.CRITICAL)
        fl.handlers = []
    except Exception:
        pass
    os.environ['TQDM_DISABLE'] = '1'


def _extract_answer(response: Any) -> str:
    text = str(response)
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for pat in [
        r'(?:total\s+cost|total|answer|result)\s*(?:is|=|:)\s*\$?([\d,]+\.?\d*)',
        r'=\s*\$?([\d,]+\.?\d*)\s*$',
        r'\*\*\$?([\d,]+\.?\d*)\*\*',
    ]:
        matches = re.findall(pat, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return '$' + matches[-1].replace(',', '')
    dollars = re.findall(r'\$([\d,]+\.?\d*)', text)
    if dollars:
        return '$' + dollars[-1].replace(',', '')
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1]
    return text.strip()[:40]


def test_m_program_robustness(cli_overrides=None):
    _suppress_noise()

    # --- Problem ---
    baseline_question = """RULES:
You are calculating total cost for a catering order.
Base price is $15 per person.
Groups of 20 or more get a 10% discount.
Weekend events have a $50 surcharge.
Delivery within 10 miles is free, beyond that costs $2 per mile.

EXAMPLES:
- 15 people, weekday, 5 miles: 15 × $15 = $225
- 25 people, weekend, 8 miles: (25 × $15 × 0.9) + $50 = $387.50
- 30 people, weekday, 15 miles: (30 × $15 × 0.9) + (5 × $2) = $415

QUESTION:
A company is ordering catering for 22 people for a Saturday event. The venue is 12 miles away. What is the total cost?"""
    ground_truth = "$351"

    # --- Config ---
    config_path = Path(__file__).parent.parent / 'config' / 'benchdrift_config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    config = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}
    if cli_overrides:
        config.update(cli_overrides)

    backend_model = config.pop('backend_model', 'granite3.3:8b')
    gen_model = config.get('gen_model', 'qwen3:8b')
    target_model = config.get('target_model', backend_model)
    top_k = config.get('top_k', 10)
    axes = config.get('use_axes', 'linguistic,referential,pragmatic,structural,constraint_targeted')
    enrich = 'on' if not config.get('no_enrich', False) else 'off'

    # --- Header ---
    print(f"\n{B}BenchDrift — M-Program Robustness Test{X}")
    print(f"{'─' * 70}")
    print(f"  Variation model  : {B}{gen_model}{X}  {D}(generates prompt variations){X}")
    print(f"  Backend model    : {B}{backend_model}{X}  {D}(Mellea m-program, model under test){X}")
    print(f"  Ground truth     : {ground_truth}")
    print(f"{'─' * 70}")

    # --- Mellea session ---
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            m = start_session(
                backend_name="ollama", model_id=backend_model,
                model_options={ModelOption.TEMPERATURE: 0.1})
    except Exception as e:
        print(f"{R}Failed to start Mellea session: {e}{X}")
        print(f"Ensure: ollama serve && ollama pull {backend_model}")
        return

    # --- M-program ---
    call_count = [0]

    def m_program(question: str) -> Any:
        call_count[0] += 1
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            response = m.instruct(question)
        return response.value if hasattr(response, 'value') else response

    # --- Progress display ---
    _out = sys.stdout

    def on_progress(current, total, status, entry):
        if status == "baseline":
            ans = _extract_answer(entry.get('variant_answer', '?'))
            ok = entry.get('correct', False)
            c = G if ok else R
            _out.write(f"\r  Baseline: {c}{ans}{X}  |  0/{total} variations...")
            _out.flush()
        elif status == "skip":
            pct = int(current / total * 100)
            _out.write(f"\r  [{pct:3d}%] {current}/{total}  {D}skip: {entry.get('variation_type','')}{X}   ")
            _out.flush()
        else:
            pct = int(current / total * 100)
            ans = _extract_answer(entry.get('variant_answer', '?'))
            c = G if status == "PASS" else R
            _out.write(f"\r  [{pct:3d}%] {current}/{total}  {c}{ans:20s}{X}  {status}   ")
            _out.flush()

    # --- Run ---
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            probes = run_benchdrift_pipeline(
                baseline_problem=baseline_question,
                ground_truth_answer=ground_truth,
                m_program_callable=m_program,
                mellea_session=m,
                answer_extractor=_extract_answer,
                config_overrides=config,
                progress_callback=on_progress,
            )
    except Exception as e:
        print(f"\n{R}Pipeline failed: {e}{X}")
        import traceback; traceback.print_exc()
        return

    _out.write("\r" + " " * 80 + "\r")
    _out.flush()

    assert probes and len(probes) > 1

    # --- Results ---
    print(f"{B}Results{X}")
    print(f"{'─' * 70}")

    bl = probes[0]
    bl_ans = _extract_answer(bl['variant_answer'])
    bl_ok = bl['correct']
    print(f"  {B}{'Baseline':30s}{X}  {(G if bl_ok else R)}{bl_ans:20s}{X}  {'PASS' if bl_ok else 'FAIL'}")

    for p in probes[1:]:
        if not p.get('is_variant'):
            continue
        ok = p.get('correct', False)
        c = G if ok else R
        label = "PASS" if ok else "FAIL"
        trans = p.get('variation_type', '?')
        ans = _extract_answer(p.get('variant_answer', '?'))
        print(f"  {trans:30s}  {c}{ans:20s}{X}  {c}{label}{X}")

    # --- Summary ---
    report = analyze_robustness(probes)
    pr = report['pass_rate']
    pc = G if pr >= 0.7 else (Y if pr >= 0.4 else R)

    print(f"{'─' * 70}")
    print(f"  Pass rate: {pc}{B}{pr:.0%}{X}  ({report['passed']}/{report['total']})"
          f"  |  baseline: {'PASS' if report['baseline_correct'] else 'FAIL'}"
          f"  |  calls: {call_count[0]}")

    # --- Save results ---
    import json
    from datetime import datetime
    output_dir = Path(__file__).parent.parent / 'logs'
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"robustness_{ts}.json"
    output_data = {
        "timestamp": ts,
        "config": {
            "backend_model": backend_model,
            "gen_model": gen_model,
            "top_k": top_k,
            "axes": axes,
            "enrich": enrich,
            "ground_truth": ground_truth,
        },
        "summary": report,
        "probes": probes,
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"  Results saved: {D}{output_file}{X}")
    print(f"{'─' * 70}\n")

    assert "error" not in report


def parse_args():
    p = argparse.ArgumentParser(description='M-program robustness test (BenchDrift)')
    p.add_argument('--backend-model', type=str, default=None,
                   help='Ollama model for Mellea m-program (default: granite3.3:8b)')
    p.add_argument('--gen-model', type=str, default=None,
                   help='Ollama model for variation generation (default: qwen3:8b)')
    p.add_argument('--top-k', type=int, default=None,
                   help='Number of ranked transformations (default: 10)')
    p.add_argument('--use-axes', type=str, default=None,
                   help='Taxonomy axes, comma-separated or "all" (default: 5 core axes)')
    p.add_argument('--no-enrich', action='store_true',
                   help='Skip LLM feature enrichment (faster)')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {}
    if args.backend_model:
        overrides['backend_model'] = args.backend_model
    if args.gen_model:
        overrides['gen_model'] = args.gen_model
    if args.top_k is not None:
        overrides['top_k'] = args.top_k
    if args.use_axes:
        overrides['use_axes'] = args.use_axes
    if args.no_enrich:
        overrides['no_enrich'] = True
    test_m_program_robustness(overrides or None)
