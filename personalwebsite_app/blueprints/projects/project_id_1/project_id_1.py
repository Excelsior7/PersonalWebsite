from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from personalwebsite_app.jinja2_env import projects_base_structure_jinja2_env


project_id_1_bp = Blueprint(name='project_id_1_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/project_id_1',
                    url_prefix="/project1");

@project_id_1_bp.route('/')
def project1():
    project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
    return render_template('project1.html', project_base_html_loader=project_base_html_loader);

