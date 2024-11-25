from celery import shared_task

@shared_task
def analyze_reviewers():
    return "analyze_reviewers"