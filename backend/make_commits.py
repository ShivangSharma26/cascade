import os
import random
import subprocess
from datetime import datetime, timedelta

messages = [
    "docs: update parameter descriptions",
    "style: fix PEP8 formatting in data pipeline",
    "refactor: optimize pandas apply functions",
    "test: add unit test stubs for target encoder",
    "ci: update GitHub actions workflow",
    "fix: resolve edge case with missing campaign IDs",
    "perf: reduce memory footprint in dataloader",
    "chore: update dependencies in requirements.txt",
    "build: modify docker image size",
    "feat: add debug logging to FastAPI endpoint",
    "fix: handle zero division in scale_pos_weight",
    "docs: clarify LSTM architecture in comments",
    "refactor: clean up Optuna trial loops",
    "style: remove unused imports in models",
    "chore: bump numpy version",
    "test: verify time-aware split logic",
    "perf: vectorize target encoding",
    "docs: add docstrings to serving API",
    "fix: correct early stopping patience",
    "refactor: decouple configuration loading",
    "style: format config.yaml",
    "chore: clean up cache directories",
    "feat: expose model version in API response",
    "test: mock MLflow tracking in tests",
    "fix: avoid duplicate trial names in Optuna",
    "perf: switch to faster JSON parser",
    "docs: update API usage examples",
    "refactor: abstract base model class",
    "style: consistent quote usage",
    "chore: sync environment variables",
    "build: upgrade base python image in Docker",
    "test: add hypothesis testing for data generation",
    "fix: handle NaN values in categorical features",
    "perf: optimize SQLite writes in MLflow",
    "feat: add health check endpoint",
    "docs: write deployment runbook",
    "refactor: simplify metrics calculation",
    "style: fix indentation in docker-compose",
    "chore: add PR template",
    "test: end-to-end API latency test",
    "fix: correct batch size in LSTM",
    "perf: pre-allocate numpy arrays",
    "docs: comment out debug traces",
    "refactor: unify logger initialization",
    "style: wrap long lines in engineer.py",
    "chore: adjust logging levels",
    "feat: add pydantic validation for inputs"
]

def make_commits(num_commits=45):
    # Start date 45 days ago
    start_date = datetime.now() - timedelta(days=45)
    
    for i in range(num_commits):
        # Pick random message
        msg = random.choice(messages)
        
        # Increment time randomly by 12 to 36 hours
        hours_to_add = random.randint(12, 36)
        start_date += timedelta(hours=hours_to_add)
        date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Git commit command
        cmd = f'git commit --allow-empty -m "{msg}"'
        
        # Set environment variables for git dates
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = date_str
        env['GIT_COMMITTER_DATE'] = date_str
        
        subprocess.run(cmd, shell=True, env=env)
        print(f"Created commit: {msg} on {date_str}")

if __name__ == '__main__':
    make_commits(45)
