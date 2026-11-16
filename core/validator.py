import re
from datetime import date
from typing import Dict, List, Optional
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

from core.graph_engine import ADIEGraphEngine
from core.models import Assumption, NodeStatus


class ValidationAlert(BaseModel):
    assumption_id: str
    statement: str
    trigger_type: str  # "TEMPORAL_EXPIRY" | "SEMANTIC_CONTRADICTION" | "NUMERIC_DRIFT"
    similarity_score: float
    details: str
    recommended_confidence: float


class HybridAssumptionValidator:
    """Neuro-Symbolic Validator combining deterministic temporal/numeric rules with local embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.55):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.negation_tokens = {
            "not", "no", "never", "increased", "exceeds", "deprecated",
            "banned", "stopped", "broken", "failed", "removed", "higher", "expensive"
        }

    def check_temporal_validity(self, engine: ADIEGraphEngine, reference_date: Optional[date] = None) -> List[ValidationAlert]:
        """Symbolic Rule 1: Checks if any assumption has passed its valid_until date."""
        today = reference_date or date.today()
        alerts: List[ValidationAlert] = []

        for assumption in engine.assumptions.values():
            if assumption.valid_until and today > assumption.valid_until and assumption.status != NodeStatus.INVALID:
                alerts.append(
                    ValidationAlert(
                        assumption_id=assumption.id,
                        statement=assumption.statement,
                        trigger_type="TEMPORAL_EXPIRY",
                        similarity_score=1.0,
                        details=f"Expired on {assumption.valid_until} (checked on {today}).",
                        recommended_confidence=0.0,
                    )
                )
        return alerts

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", text)]

    def evaluate_incoming_fact(
        self,
        engine: ADIEGraphEngine,
        new_fact: str,
        auto_propagate: bool = True,
    ) -> List[ValidationAlert]:
        """Neural + Symbolic Check: Matches an incoming observation against stored assumptions."""
        assumptions = list(engine.assumptions.values())
        if not assumptions:
            return []

        statements = [a.statement for a in assumptions]
        fact_emb = self.model.encode(new_fact, convert_to_tensor=True)
        stmt_embs = self.model.encode(statements, convert_to_tensor=True)
        cosine_scores = util.cos_sim(fact_emb, stmt_embs)[0]

        fact_numbers = self._extract_numbers(new_fact)
        fact_words = set(re.findall(r"\w+", new_fact.lower()))
        has_conflict_cue = bool(fact_words & self.negation_tokens)

        alerts: List[ValidationAlert] = []

        for idx, assumption in enumerate(assumptions):
            sim = float(cosine_scores[idx])
            if sim < self.similarity_threshold:
                continue

            old_numbers = self._extract_numbers(assumption.statement)
            numeric_changed = bool(old_numbers and fact_numbers and old_numbers != fact_numbers)

            if numeric_changed:
                alert = ValidationAlert(
                    assumption_id=assumption.id,
                    statement=assumption.statement,
                    trigger_type="NUMERIC_DRIFT",
                    similarity_score=round(sim, 4),
                    details=f"Matched (sim={sim:.2f}). Value shifted from {old_numbers} to {fact_numbers}.",
                    recommended_confidence=0.1,
                )
                alerts.append(alert)
                if auto_propagate:
                    engine.invalidate_assumption(assumption.id, new_confidence=0.1)

            elif has_conflict_cue:
                alert = ValidationAlert(
                    assumption_id=assumption.id,
                    statement=assumption.statement,
                    trigger_type="SEMANTIC_CONTRADICTION",
                    similarity_score=round(sim, 4),
                    details=f"Matched (sim={sim:.2f}) with contradiction signal in: '{new_fact}'.",
                    recommended_confidence=0.2,
                )
                alerts.append(alert)
                if auto_propagate:
                    engine.invalidate_assumption(assumption.id, new_confidence=0.2)

        return alerts