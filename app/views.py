# from flask import Blueprint, render_template, redirect, request
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
