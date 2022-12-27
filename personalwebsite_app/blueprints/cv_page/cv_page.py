from flask import Blueprint, render_template, redirect, url_for
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

@cv_bp.route('/pdf')
def cvRouting():
    return redirect(url_for('cv_page_bp.static', filename='resume.pdf'));