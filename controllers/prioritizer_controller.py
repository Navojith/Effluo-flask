from flask import Blueprint, request, jsonify
import logging
from app.pr_prioritizer.inference_pipeline.inference import PRPrioritizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Blueprint
pr_prioritizer_bp = Blueprint('pr_prioritizer', __name__)

# Initialize the prioritizer at the module level for reuse
prioritizer = None

def get_prioritizer():
    """Lazy-load the PR Prioritizer model"""
    global prioritizer
    if prioritizer is None:
        logger.info("Initializing PR Prioritizer model")
        prioritizer = PRPrioritizer('./app/models/pr_priority_model.pkl')
    return prioritizer


def pr_prioritizer_controller(prs_data):
    """
    Direct controller function for PR prioritization
    
    Args:
        prs_data (list): List of PR dictionaries with features
        
    Returns:
        list: Prioritized PRs with predicted priorities and confidence scores
    """
    # Initialize or get prioritizer
    prioritizer = get_prioritizer()
    
    # Generate predictions
    results = prioritizer.predict_priorities(prs_data)
    
    return results