"""
Public API Layer for the Causal Neuro-Symbolic Diagnosis framework.

This module provides a clean, domain-agnostic wrapper over the internal
diagnosis, causal, refutation, SCM, and counterfactual reasoning engines.
"""

from typing import Dict, Any, Optional
from cnsd.config import load_config
from cnsd.diagnosis.system import CNSD as InternalEngine


class CNSD:
    """
    Public Framework Entry Point.
    
    Initializes the end-to-end framework dynamically from a YAML configuration.
    """

    def __init__(self, config: str = "cnsd_config.yaml"):
        """
        Args:
            config: Path to the YAML configuration file defining the domain.
        """
        self.config = load_config(config)
        
        # We pass the domain-agnostic config to the internal engine
        # (Assuming the internal refactoring will eventually pipe self.config deeper)
        self._engine = InternalEngine()
        self._fitted = False

    def _ensure_fitted(self, data):
        """Internal helper to lazy-fit the pipeline."""
        if not self._fitted:
            self._engine.fit(data)
            self._fitted = True

    def diagnose(self, data) -> Any:
        """
        Diagnosis API: Executes Layer 1 (CNN) and Layer 2 (Symbolic Physics)
        to identify the root cause of the fault.

        Args:
            data: Dataset object containing time-series signals and operating conditions.
            
        Returns:
            DiagnosisReport: A structured report containing predicted faults, 
                             confidence scores, and actionable maintenance metadata.
        """
        self._ensure_fitted(data)
        return self._engine.diagnose(data)

    def explain(self, data) -> Any:
        """
        Causal Analysis API: Exposes Pearl's Rung-2 (Intervention) reasoning.
        Evaluates the causal effect of the operating condition on the fault.

        Args:
            data: Dataset object containing signals and conditions.
            
        Returns:
            Intervention effect summary containing estimated causal effects
            and identified confounders.
        """
        self._ensure_fitted(data)
        return self._engine.condition_effect(data)

    def what_if(self, data, intervention: Dict[str, float], unit_index: int = 0) -> Any:
        """
        Counterfactual API: Exposes Pearl's Rung-3 reasoning.
        Predicts what the fault status *would have been* if the operating condition
        had been different.

        Args:
            data: Dataset object.
            intervention: A dictionary mapping the condition variable to a hypothetical value.
                          (e.g., {"load": 0.8})
            unit_index: The specific machine index in the dataset to run the counterfactual on.
            
        Returns:
            Counterfactual prediction result.
        """
        self._ensure_fitted(data)
        
        # Extract the hypothetical condition value
        # For a single operating condition, we grab the first value in the dict
        condition_cf = list(intervention.values())[0] if intervention else 0.0
        
        return self._engine.what_if(data, unit_index=unit_index, condition_cf=condition_cf)

    def scm_analysis(self, data) -> Any:
        """
        SCM API: Provides direct access to the generated Structural Causal Model (SCM).

        Args:
            data: Dataset object.
            
        Returns:
            The underlying DoWhy/NetworkX Structural Causal Model object containing
            structural relationships and graph information.
        """
        self._ensure_fitted(data)
        return self._engine.scm
