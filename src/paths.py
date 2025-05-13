"""
This file contains paths to key directories we'll use in the project
"""
from pathlib import Path

# path to the GitHub repository
repo_dir = Path(__file__).resolve().parent.parent

# directory where we keep jinja templates
template_dir = repo_dir / 'templates'

# directory for static elements used in flask
static_dir = repo_dir / 'static'

# directory where we keep image files
image_dir = static_dir / 'images'

# directory where we store results
results_dir = repo_dir / 'results'

# directory where we store reporting code
reporting_dir = repo_dir / 'reporting'

# directory where we store RMarkdown templates
report_templates = reporting_dir / 'templates'

# directory where we store reports (e.g., pdfs)
reports_dir = repo_dir / 'reports'

# create local directories if they do not exist
results_dir.mkdir(exist_ok = True)