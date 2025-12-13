"""BenchDrift-Mellea integration toolkit for robustness testing of Mellea programs."""

import json
import logging
import tempfile
from typing import List, Dict, Any, Callable, Optional, Tuple

from benchdrift.pipeline.unified_batched_pipeline_semantic import UnifiedBatchedPipeline
from benchdrift.eval.llm_answer_matcher import LLMAnswerMatcher
from mellea import MelleaSession
from mellea.backends.types import ModelOption

# Import the enhanced adapter
from mellea_contribs.tools.mellea_model_client_adapter import MelleaModelClientAdapter

logger = logging.getLogger(__name__)

# --- Core API Functions ---

def run_benchdrift_pipeline(
    baseline_problem: str,
    ground_truth_answer: str,
    m_program_callable: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    mellea_session: Optional[MelleaSession] = None,
    response_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    generation_model: Optional[str] = None,
    answer_extractor: Optional[Callable[[Any], str]] = None,
    max_workers: int = 4,
    config_overrides: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Execute 3-stage BenchDrift pipeline (variations → responses → evaluation)."""
    # Validate m-program parameters
    if (m_program_callable is None) != (mellea_session is None):
        raise ValueError(
            "Both m_program_callable and mellea_session must be provided together. "
            f"Got: m_program={m_program_callable is not None}, session={mellea_session is not None}"
        )

    # Create input data
    input_problem_data = [{"problem": baseline_problem, "answer": ground_truth_answer}]

    # Initialize config
    if config_overrides is None:
        config_overrides = {}

    # Prepare temporary files
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json") as temp_input_file:
        json.dump(input_problem_data, temp_input_file)
        temp_input_filename = temp_input_file.name

    # Create output file path but DON'T create the file - BenchDrift will create it
    temp_dir = tempfile.gettempdir()
    temp_output_filename = os.path.join(temp_dir, f"benchdrift_output_{os.getpid()}_{id(input_problem_data)}.json")

    try:
        # Build pipeline config
        config_semantic = {
            'unified_file': temp_output_filename,
            'input_problems': temp_input_filename,
            'batch_size': 2,
            'max_workers': 4,
            'client_type': 'rits',
            'model_name': generation_model or 'phi-4',
            'judge_model': judge_model or 'llama_3_3_70b',
            'response_model': response_model or 'granite-3-3-8b',
            'response_client_type': 'rits',
            'use_llm_judge': True,
            'rectify_invalid': True,
            'max_model_len': 5000,
            'max_new_tokens': 1000,
            'embedding_model': 'all-MiniLM-L6-v2',
            'semantic_threshold': 0.35,
            'use_cagrad_dependencies': False,
            'use_generic': True,
            'use_cluster_variations': True,
            'use_persona': False,
            'use_long_context': False,
            'verbose': False,
        }

        # Apply user overrides
        if config_overrides:
            config_semantic.update(config_overrides)

        # If m-program provided: create adapter and use it as response model
        if m_program_callable is not None:
            logger.info("✅ M-program provided: Creating MelleaModelClientAdapter")
            adapter = MelleaModelClientAdapter(
                m_program_callable=m_program_callable,
                mellea_session=mellea_session,
                answer_extractor=answer_extractor,
                max_workers=max_workers
            )
            # Override response_model to use the adapter
            config_semantic['response_model'] = adapter
            logger.info("✅ Adapter set as response_model for Stage 2")
        else:
            logger.info(f"📦 Using standard model: {config_semantic['response_model']}")

        # Execute pipeline stages
        logger.info("\n🚀 Running BenchDrift Pipeline...")
        logger.info("📝 Stage 1: Generating semantic variations...")
        pipeline = UnifiedBatchedPipeline(config_semantic)
        pipeline.stage1_generate_variations_batched()

        logger.info("✅ Validating variations...")
        pipeline.stage_validation()

        logger.info("🔄 Stage 2: Generating model responses...")
        if m_program_callable is not None:
            logger.info("   (Using m-program via MelleaModelClientAdapter)")
        pipeline.stage2_generate_responses()

        logger.info("📊 Stage 3: Evaluating drift metrics...")
        pipeline.stage3_add_evaluation_metrics()

        logger.info("\n🎉 BenchDrift pipeline completed successfully!")

        # Load and return results
        with open(temp_output_filename, 'r') as f:
            results_data = json.load(f)

        logger.info(f"📊 Generated {len(results_data)} total entries")
        variant_count = sum(1 for r in results_data if r.get('is_variant'))
        logger.info(f"   - Variants: {variant_count}")
        logger.info(f"   - Baselines: {len(results_data) - variant_count}")

        print(f"\n📁 Result JSON saved to: {temp_output_filename}", flush=True)

        return results_data

    finally:
        # Cleanup temporary files
        try:
            os.remove(temp_input_filename)
            # Keep output file for inspection - don't delete temp_output_filename
            # os.remove(temp_output_filename)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")


# ===== RESULT EXTRACTION FUNCTIONS (FROM PIPELINE OUTPUT) =====
#
# These functions ONLY extract and analyze data from the result JSON
# produced by BenchDrift's 3 stages. They do NOT call any evaluation logic.
# Everything is read directly from the output file.
#
# Key Insight:
# - BenchDrift Stage 1: Generates variations
# - BenchDrift Stage 2: Gets responses from m-program (via adapter)
# - BenchDrift Stage 3: LLM judge evaluates and writes drift flags to JSON
#
# Mellea-contribs: Just reads what BenchDrift wrote. No re-evaluation.

def analyze_robustness_from_probes(probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze robustness metrics from pipeline results (pass rate, drift, stability)."""
    # Filter to variants only (Stage 3 already evaluated these)
    variant_probes = [p for p in probes if p.get('is_variant')]

    if not variant_probes:
        logger.warning("⚠️  No variants found in probes")
        return {
            "error": "No variants found",
            "overall_pass_rate": 0.0,
            "total_variants": 0
        }

    # READ from result JSON - no computation
    # These fields were written by Stage 3
    correct = sum(1 for p in variant_probes if p.get('variant_matches_ground_truth'))
    incorrect = len(variant_probes) - correct

    # Drift counts (from Stage 3 fields)
    positive_drift_count = sum(1 for p in variant_probes if p.get('positive_drift'))
    negative_drift_count = sum(1 for p in variant_probes if p.get('negative_drift'))
    no_drift_count = len(variant_probes) - positive_drift_count - negative_drift_count

    # By variation type (metadata from Stage 1)
    by_type = {}
    for probe in variant_probes:
        var_type = probe.get('variation_type', 'unknown')
        if var_type not in by_type:
            by_type[var_type] = {'total': 0, 'correct': 0}
        by_type[var_type]['total'] += 1
        # 'correct' = variant answered correctly (from Stage 3)
        if probe.get('variant_matches_ground_truth'):
            by_type[var_type]['correct'] += 1

    by_type_rates = {
        var_type: stats['correct'] / stats['total']
        for var_type, stats in by_type.items()
    }

    # Stability: Compare baseline vs variant (both from Stage 3)
    baseline_consistent = sum(
        1 for p in variant_probes
        if p.get('baseline_matches_ground_truth') == p.get('variant_matches_ground_truth')
    )

    return {
        "overall_pass_rate": correct / len(variant_probes),
        "total_variants": len(variant_probes),
        "pass_count": correct,
        "fail_count": incorrect,
        "drift_analysis": {
            "positive_drift_count": positive_drift_count,
            "negative_drift_count": negative_drift_count,
            "no_drift_count": no_drift_count,
            "positive_drift_rate": positive_drift_count / len(variant_probes),
            "negative_drift_rate": negative_drift_count / len(variant_probes)
        },
        "by_variation_type": by_type_rates,
        "stability_metrics": {
            "baseline_consistent_count": baseline_consistent,
            "baseline_consistency_rate": baseline_consistent / len(variant_probes)
        }
    }


def extract_repair_candidates(probes: List[Dict[str, Any]],
                             baseline_problem: str) -> List[str]:
    """Extract variants with positive drift (work when baseline fails)."""
    # Check if baseline failed
    baseline_entry = next((p for p in probes if p.get('is_baseline')), None)
    if not baseline_entry or baseline_entry.get('baseline_matches_ground_truth'):
        logger.info("ℹ️  Baseline passed - no repair needed")
        return []

    logger.info("⚠️  Baseline failed - searching for repair candidates...")

    # Find variants with positive drift (they work when baseline doesn't)
    repair_candidates = []
    for probe in probes:
        if (probe.get('is_variant') and
            probe.get('positive_drift')):
            modified_problem = probe.get('modified_problem')
            if modified_problem:
                repair_candidates.append(modified_problem)

    logger.info(f"✅ Found {len(repair_candidates)} repair candidates")
    return repair_candidates


def extract_replacement_instructions(probes: List[Dict[str, Any]]) -> List[str]:
    """Extract variants that answer correctly (validated alternatives)."""
    alternatives = []
    for probe in probes:
        if (probe.get('is_variant') and
            probe.get('variant_matches_ground_truth')):
            modified_problem = probe.get('modified_problem')
            if modified_problem:
                alternatives.append(modified_problem)

    logger.info(f"✅ Found {len(alternatives)} working alternative phrasings")
    return alternatives


def evaluate_program_robustness(
    probes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract robustness metrics from pipeline results (pass rate, failures, drift)."""
    logger.info(f"\n🔬 Analyzing Mellea program robustness from {len(probes)} probes...")

    # Filter to variants (stage 3 already evaluated these)
    variant_probes = [p for p in probes if p.get('is_variant')]
    if not variant_probes:
        error_msg = "No variants found in probes"
        logger.error(f"❌ {error_msg}")
        return {"error": error_msg, "pass_rate": 0.0, "total_probes": 0}

    # COUNT results directly from result JSON (Stage 3 fields)
    pass_count = 0
    failures = []
    drift_summary = {"positive": 0, "negative": 0, "no_change": 0}

    for i, probe in enumerate(variant_probes):
        # Read from Stage 3 field
        variant_correct = probe.get('variant_matches_ground_truth')

        if variant_correct:
            pass_count += 1
        else:
            # Track failures (already evaluated by Stage 3)
            failures.append({
                "probe_index": i,
                "problem": probe.get('modified_problem'),
                "expected_answer": probe.get('ground_truth_answer'),
                "variant_answer": probe.get('variant_answer'),
            })

        # Read drift flags from Stage 3
        if probe.get('positive_drift'):
            drift_summary["positive"] += 1
        elif probe.get('negative_drift'):
            drift_summary["negative"] += 1
        else:
            drift_summary["no_change"] += 1

    pass_rate = pass_count / len(variant_probes)

    report = {
        "pass_rate": pass_rate,
        "pass_count": pass_count,
        "fail_count": len(failures),
        "total_probes": len(variant_probes),
        "failures": failures,
        "drift_summary": drift_summary
    }

    # Log summary
    logger.info(f"📊 Robustness Report (from result JSON):")
    logger.info(f"   Pass Rate: {pass_rate:.2%} ({pass_count}/{len(variant_probes)})")
    logger.info(f"   Drift Summary (from Stage 3):")
    logger.info(f"     - Positive (improved): {drift_summary['positive']}")
    logger.info(f"     - Negative (worsened): {drift_summary['negative']}")
    logger.info(f"     - No change: {drift_summary['no_change']}")

    return report
