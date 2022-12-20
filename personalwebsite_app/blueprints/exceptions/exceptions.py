from flask import Blueprint, render_template, request, redirect, url_for
from personalwebsite_app.jinja2_env import base_structure_jinja2_env

exceptions_bp = Blueprint(name='exceptions_page_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/exceptions',
                    url_prefix="/error");

# see - https://flask.palletsprojects.com/en/2.2.x/errorhandling/#unhandled-exceptions
def internalServerError():
    base_html_loader = base_structure_jinja2_env.get_template("base.html");
    return render_template('500.html', base_html_loader=base_html_loader);


def pageNotFound404(e):
    if request.path.startswith('/cv/'):
        return redirect(url_for('cv_page_bp.cv'));
    elif request.path.startswith('/home/'):
        return redirect(url_for('home_page_bp.home'));
    elif request.path.startswith('/project1/'):
        return redirect(url_for('project_id_1_bp.project1'));
    elif request.path.startswith('/project2/'):
        return redirect(url_for('project_id_2_bp.project2'));
    elif request.path.startswith('/projects/'):
        return redirect(url_for('projects_page_bp.projects'));
    else:
        return redirect(url_for('home_page_bp.home'));
