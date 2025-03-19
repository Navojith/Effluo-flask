# from flask import Blueprint, jsonify, request
from controllers.prioritizer_controller import pr_prioritizer_controller
from app.controllers import mcr_controller
from flask import Blueprint, request, jsonify
from .reviewer_assignment import analyze_reviewers

main = Blueprint("main", __name__)


@main.route("/health", methods=["GET", "POST"])
def health():
    return "OK"


@main.route("/analyze-reviewers", methods=["GET", "POST"])
def test():
    return analyze_reviewers()


@main.route("/prioritize-pr", methods=["POST"])
def predict_pr_priorities():
    """
    Controller endpoint for predicting PR priorities

    Expected JSON input format:
    {
        "pull_requests": [pip install xgboost
            {
                "id": "PR123",
                "title": "Fix memory leak in service",
                "body": "This PR addresses the memory leak issue...",
                "author_association": "CONTRIBUTOR",
                "comments": 5,
                "additions": 120,
                "deletions": 50,
                "changed_files": 3
            },
            ...
        ]
    }

    Returns:
        JSON with predictions for each PR
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data or "pull_requests" not in data:
            return (
                jsonify(
                    {"error": 'Invalid request format. Expected "pull_requests" array.'}
                ),
                400,
            )

        prs_data = data["pull_requests"]
        print(prs_data)

        if not prs_data or not isinstance(prs_data, list):
            return (
                jsonify(
                    {
                        "error": "Empty or invalid pull_requests data. Expected non-empty array."
                    }
                ),
                400,
            )

        # Process the PRs and predict priorities
        print(f"Processing {len(prs_data)} pull requests")
        results = pr_prioritizer_controller(prs_data)

        # Return predictions
        return jsonify({"status": "success", "predictions": results})

    except Exception as e:
        print(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500


@main.route("/mcr", methods=["POST"])
def mcr():
    try:
        data = request.json
        base_code = data.get("base_code", "")
        branch_a_code = data.get("branch_a_code", "")
        branch_b_code = data.get("branch_b_code", "")

        resolved_code = mcr_controller.mcr_controller(
            base_code, branch_a_code, branch_b_code
        )

        return jsonify({"status": "success", "resolved_code": resolved_code})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
