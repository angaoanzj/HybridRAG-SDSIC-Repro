"""Vocabulary interventions around the retained title-selection implementations."""
from copy import deepcopy
import os
from cue_runtime import get_matcher
from query_expansion import QueryExpander
from frozen_entity_variants import VariantExpander as EntityExpander
from frozen_order_variants import VariantExpander as OrderExpander
from frozen_core_variants import VariantExpander as CoreExpander

POLICIES = (
    "hybrid_rrf", "first_two_valid_titles", "evidence_feedback",
    "score2_old24", "score2_none", "score2_u24", "score2_morph",
    "no_second_branch", "one_target", "bridge_expand", "score_two_reverse",
    "score1_u24", "family_u24", "family_morph", "reverse_u24",
    "no_source_u24", "family_no_cue", "no_residual_u24", "repeat_q0",
)
VOCABULARY_MODES = {
    "score2_old24": "old24", "score2_none": "none",
    "score2_u24": "unsupervised", "score2_morph": "morphology",
    "score1_u24": "unsupervised", "family_u24": "unsupervised",
    "family_morph": "morphology", "reverse_u24": "unsupervised",
    "no_source_u24": "unsupervised", "family_no_cue": "unsupervised",
    "no_residual_u24": "unsupervised",
}
VARIANT_VERSION = "external_plaintext_cues_fresh_evaluation_20261108_v1"
DELEGATE_POLICIES = {"score1_u24": "one_target", "family_u24": "bridge_expand",
                     "family_morph": "bridge_expand", "reverse_u24": "no_second_branch",
                     "no_source_u24": "no_source_priority", "family_no_cue": "no_relation_cue",
                     "no_residual_u24": "no_residual", "repeat_q0": "hybrid_rrf"}

class VariantExpander:
    def __init__(self, docs=None, policy="bridge_expand", source_top_n=5, title_lookup=None):
        if policy not in POLICIES or source_top_n != 5:
            raise ValueError("Condition outside the experiment protocol")
        self.policy = policy
        mode = VOCABULARY_MODES.get(policy, "synonyms")
        delegate_policy = DELEGATE_POLICIES.get(policy, policy)
        delegate_class = EntityExpander
        if policy in {"score2_old24", "score2_none", "score2_u24", "score2_morph", "score_two_reverse"}:
            delegate_policy = "no_second_branch"
        if policy == "first_two_valid_titles":
            delegate_class = OrderExpander
        if policy in {"evidence_feedback", "no_source_u24", "family_no_cue", "no_residual_u24"}:
            delegate_class = CoreExpander
        self.delegate = delegate_class(docs=docs, policy=delegate_policy,
                                       source_top_n=5, title_lookup=title_lookup)
        self.delegate.cue_matcher = get_matcher(mode)
        self.title_lookup = self.delegate.title_lookup
        self.params = deepcopy(self.delegate.params)
        self.params.update({"variant_version": VARIANT_VERSION, "experiment_policy": policy,
                            "cue_mode": mode, "cue_config_sha256": os.environ["HYBRIDRAG_CUE_SHA256"]})

    def bridge_query(self, question_text, initial_ids, doc_lookup):
        result = self.delegate.bridge_query(question_text, initial_ids, doc_lookup)
        if self.policy not in {"score_two_reverse", "reverse_u24"}:
            return result
        result = deepcopy(result)
        provenance = result["provenance"]
        reference = deepcopy(provenance["entities"])
        reordered = list(reversed(reference)) if len(reference) == 2 else reference
        provenance["entities"] = reordered
        provenance["reviewer_order_control"] = {
            "reference_policy": "no_second_branch", "reference_entities": reference,
            "reference_query": result["query"], "reversed_two_title_order": len(reference) == 2,
            "same_selected_target_set": True, "residual_keywords_unchanged": True,
        }
        if reordered:
            result["query"] = " ".join([item["mentioned_title"] for item in reordered] +
                                       provenance["residual_keywords"])
        return result

def variant_parameters(policy="bridge_expand", source_top_n=5):
    return VariantExpander(docs=[], policy=policy, source_top_n=source_top_n).params
