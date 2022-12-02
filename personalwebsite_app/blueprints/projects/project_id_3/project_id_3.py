from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound


project_id_3_bp = Blueprint(name='project_id_3_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/project_id_3',
                    url_prefix="/project3");

@project_id_3_bp.route('/')
def project3():
    return render_template('project3.html');