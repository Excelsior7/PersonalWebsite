from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from personalwebsite_app.jinja2_env import base_structure_jinja2_env

cv_bp = Blueprint(name='cv_page_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/cv_page',
                    url_prefix="/cv");


@cv_bp.route('/')
def cv():
    base_html_loader = base_structure_jinja2_env.get_template("base.html");
    return render_template('cv.html', base_html_loader=base_html_loader);