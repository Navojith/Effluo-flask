from flask import Blueprint,jsonify, render_template, redirect, request
from .reviewer_assignment import analyze_reviewers
from app.db_models import User
from app.db import get_session

main = Blueprint('main', __name__)

session = get_session()

@main.route('/health', methods=['GET', 'POST'])
def health():
    return "OK"

@main.route('/analyze-reviewers', methods=['GET', 'POST'])
def test():
    return analyze_reviewers()

@main.route('/user', methods=['GET', 'POST'])
def get_users():
    users = session.query(User).all()
    return jsonify([user.to_dict() for user in users])