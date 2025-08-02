import os
import markdown
from flask import Blueprint, render_template, send_file, abort
from pathlib import Path

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@main_bp.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@main_bp.route('/dissertation')
def dissertation():
    """Dissertation viewer"""
    try:
        # Get the dissertation markdown file
        dissertation_path = Path(__file__).parent.parent.parent / "FraudGuard_MSc_Dissertation.md"
        
        if not dissertation_path.exists():
            abort(404, "Dissertation file not found")
        
        # Read and convert markdown to HTML
        with open(dissertation_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to HTML with extensions
        html_content = markdown.markdown(
            markdown_content, 
            extensions=['toc', 'tables', 'fenced_code', 'footnotes']
        )
        
        return render_template('dissertation.html', content=html_content)
    
    except Exception as e:
        abort(500, f"Error loading dissertation: {str(e)}")

@main_bp.route('/dissertation/download')
def download_dissertation():
    """Download dissertation as markdown file"""
    try:
        dissertation_path = Path(__file__).parent.parent.parent / "FraudGuard_MSc_Dissertation.md"
        
        if not dissertation_path.exists():
            abort(404, "Dissertation file not found")
        
        return send_file(
            dissertation_path,
            as_attachment=True,
            download_name="FraudGuard_MSc_Dissertation.md",
            mimetype="text/markdown"
        )
    
    except Exception as e:
        abort(500, f"Error downloading dissertation: {str(e)}")