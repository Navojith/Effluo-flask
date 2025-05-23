from app.merge_conflict_resolver.inference_pipeline.inference import (
    HierarchicalMergeConflictResolver,
)


def hierarchical_mcr_controller(base_code, a_code, b_code):
    resolver = HierarchicalMergeConflictResolver(
        "./app/models/best_token_model.pt",
        "./app/models/best_syntax_token_model.pt",
    )

    resolved_code = resolver.resolve_conflict(base_code, a_code, b_code)

    return resolved_code
