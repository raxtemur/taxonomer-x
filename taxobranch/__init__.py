"""Public TaxoBranch API for exact taxonomy decoding and sensitivity analysis."""

from src.symbolic.maximum_branching import (
    ROOT,
    BranchCandidate,
    BranchingResult,
    FeasibilityCertificate,
    verify_feasibility,
    maximum_branching,
)
from src.symbolic.counterfactual_certificates import (
    DecisionCertificate,
    certificate_features,
    exact_decision_certificates,
)

# Public terminology used by the paper. Historical names remain aliases so
# released experiment manifests and imports stay reproducible.
decode_taxonomy = maximum_branching
DecisionSensitivity = DecisionCertificate
exact_decision_sensitivities = exact_decision_certificates
sensitivity_features = certificate_features

__all__ = [
    "ROOT",
    "BranchCandidate",
    "BranchingResult",
    "FeasibilityCertificate",
    "DecisionSensitivity",
    "decode_taxonomy",
    "exact_decision_sensitivities",
    "sensitivity_features",
    "verify_feasibility",
]

__version__ = "0.1.0"
