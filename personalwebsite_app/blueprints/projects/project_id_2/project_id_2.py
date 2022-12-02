from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound


project_id_2_bp = Blueprint(name='project_id_2_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/project_id_2',
                    url_prefix="/project2");

@project_id_2_bp.route('/')
def project2():
    return render_template('project2.html');