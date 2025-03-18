from flask import Flask
from flask_cors import CORS

from .views import main
from .celery import make_celery


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config["CELERY_CONFIG"] = {
        "broker_url": "redis://localhost:6379",
        "result_backend": "redis://localhost:6379",
        "beat_schedule": {
            "Analyze reviewers": {
                "task": "app.reviewer_assignment.reviewer_assignment.analyze_reviewers",
                "schedule": 60,
            }
        },
    }

    celery = make_celery(app)
    celery.set_default()

    app.register_blueprint(main)

    return app, celery
