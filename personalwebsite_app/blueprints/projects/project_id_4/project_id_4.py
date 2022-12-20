from flask import Blueprint, render_template


project_id_4_bp = Blueprint(name='project_id_4_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/project_id_4',
                    url_prefix="/project4");

@project_id_4_bp.route('/')
def project4():
    return render_template('project4.html');