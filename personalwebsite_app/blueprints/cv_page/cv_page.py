from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

cv_bp = Blueprint(name='cv_page_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/cv_page',
                    url_prefix="/cv");

@cv_bp.route('/')
def cv():
    return render_template('cv.html');