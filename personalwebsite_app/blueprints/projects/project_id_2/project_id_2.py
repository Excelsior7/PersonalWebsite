from flask import Blueprint, render_template, request, abort
from jinja2 import TemplateNotFound
from personalwebsite_app.jinja2_env import projects_base_structure_jinja2_env


project_id_2_bp = Blueprint(name='project_id_2_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/project_id_2',
                    url_prefix="/project2");

@project_id_2_bp.route('/')
def project2():
    project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
    return render_template('project2.html', 
                            project_base_html_loader=project_base_html_loader, 
                            en_to_fr_display="block", 
                            fr_to_en_display="none");


@project_id_2_bp.route('/en', methods=['POST'])
def translateEnglishToFrench():
    en_input = request.form['input'];
    en_to_fr_translation = en_input + "english";

    project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
    return render_template('project2.html', 
                            project_base_html_loader=project_base_html_loader,
                            en_input = en_input,
                            en_to_fr_translation=en_to_fr_translation,
                            en_to_fr_display="block", 
                            fr_to_en_display="none");


@project_id_2_bp.route('/fr', methods=['POST'])
def translateFrenchToEnglish():
    fr_input = request.form['input'];
    fr_to_en_translation = fr_input + "french";

    project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
    return render_template('project2.html', 
                            project_base_html_loader=project_base_html_loader,
                            fr_input=fr_input,
                            fr_to_en_translation=fr_to_en_translation,
                            en_to_fr_display="none", 
                            fr_to_en_display="block");


