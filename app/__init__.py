from flask import Flask
from celery.schedules import crontab

from .views import main
from .celery import make_celery

def create_app():
    app = Flask(__name__)

    app.config["CELERY_CONFIG"] = {"broker_url": "redis://redis", "result_backend": "redis://redis", "beat_schedule": {
        "Analyze reviewers" : {
            "task": "app.reviewer_assignment.reviewer_assignment.analyze_reviewers",
            "schedule": 60,
            # "schedule": crontab(hour=2, minute=25, day_of_week=1),
            #"args": (1, 2)
        }
    }}

    celery = make_celery(app)
    celery.set_default()

    app.register_blueprint(main)

    return app, celery