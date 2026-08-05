"""Adapter bridge between Mellea m-programs and BenchDrift's BaseModelClient interface."""

import logging
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# BenchDrift imports
from benchdrift.models.model_client import BaseModelClient

# Mellea imports
from mellea import MelleaSession
from mellea.backends.types import ModelOption

logger = logging.getLogger(__name__)


class MelleaModelClientAdapter(BaseModelClient):
    """Adapts Mellea m-programs to BenchDrift's BaseModelClient interface."""

    def __init__(self,
                 m_program_callable: Callable[[str, Dict[str, Any]], Any],
                 mellea_session: MelleaSession,
                 answer_extractor: Optional[Callable[[Any], str]] = None,
                 max_workers: int = 4):
        """Initialize adapter with m-program, session, and optional answer extractor."""
        self.m_program = m_program_callable
        self.session = mellea_session
        self.answer_extractor = answer_extractor or self._default_answer_extractor
        self.max_workers = max_workers

        logger.debug(
            f"✅ MelleaModelClientAdapter initialized with "
            f"m_program={m_program_callable.__name__}, "
            f"max_workers={max_workers}"
        )

    def get_model_response(self,
                          system_prompts: List[str],
                          user_prompts: List[str],
                          max_new_tokens: int = 1000,
                          temperature: float = 0.1,
                          **kwargs) -> List[str]:
        """Generate batch responses for prompts using m-program with parallel processing."""
        logger.debug(
            f"🔄 get_model_response() called: batch_size={len(user_prompts)}, "
            f"max_workers={self.max_workers}"
        )

        # Build full prompts (combine system + user)
        full_prompts = [
            self._build_full_prompt(sys_prompt, usr_prompt)
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts)
        ]

        # Process in parallel using ThreadPoolExecutor
        responses = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_idx = {
                executor.submit(self._call_m_program, prompt): i
                for i, prompt in enumerate(full_prompts)
            }

            # Collect results in order
            results_by_idx = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    response = future.result()
                    extracted_answer = self.answer_extractor(response)
                    results_by_idx[idx] = extracted_answer
                    logger.debug(f"   ✅ Processed prompt {idx + 1}/{len(full_prompts)}")
                except Exception as e:
                    error_msg = f"[ERROR: {str(e)[:100]}]"
                    results_by_idx[idx] = error_msg
                    logger.error(f"   ❌ Failed to process prompt {idx + 1}: {e}")

            # Sort by original index to maintain order
            responses = [results_by_idx[i] for i in range(len(full_prompts))]

        logger.debug(f"✅ Batch processing complete: {len(responses)} responses")
        return responses

    def get_single_response(self,
                           system_prompt: str,
                           user_prompt: str,
                           max_new_tokens: int = 1000,
                           temperature: float = 0.1,
                           **kwargs) -> str:
        """Generate single response by delegating to batch interface."""
        logger.debug(f"🔄 get_single_response() called")

        responses = self.get_model_response(
            [system_prompt],
            [user_prompt],
            max_new_tokens,
            temperature,
            **kwargs
        )

        result = responses[0] if responses else "[ERROR: No response]"
        logger.debug(f"✅ Single response: {result[:80]}...")
        return result

    def _call_m_program(self, prompt: str) -> Any:
        """Invoke m-program with given prompt."""
        return self.m_program(prompt)

    def _build_full_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Combine system and user prompts into single prompt string."""
        if system_prompt and system_prompt.strip():
            return f"{system_prompt}\n\n{user_prompt}"
        return user_prompt


    @staticmethod
    def _default_answer_extractor(response: Any) -> str:
        """Extract answer from m-program response (try .value, fallback to str)."""
        if hasattr(response, 'value'):
            return str(response.value)
        return str(response)
