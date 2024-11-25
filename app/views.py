from flask import Blueprint, render_template, redirect, request
from .reviewer_assignment import analyze_reviewers

main = Blueprint('main', __name__)

@main.route('/health', methods=['GET', 'POST'])
def health():
    return "OK"

@main.route('/analyze-reviewers', methods=['GET', 'POST'])
def test():
    return analyze_reviewers()