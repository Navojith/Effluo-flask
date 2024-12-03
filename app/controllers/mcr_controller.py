from app.merge_conflict_resolver.inference_pipeline.inference import MergeConflictResolver


def mcr_controller(base_code, a_code, b_code):
    """
    Merge conflict resolution controller

    Args:
        base_code (str): Base code
        a_code (str): Code from branch A
        b_code (str): Code from branch B

    Returns:
        str: Resolved code
    """
    # Initialize model
    resolver = MergeConflictResolver('./app/merge_conflict_resolver/models/best_model.pt')

    # Preprocess input
    preprocessed_input = resolver.preprocess_merge_conflict(base_code, a_code, b_code)

    # Generate resolution
    resolved_code = resolver.generate_resolution(preprocessed_input)

    return resolved_code
