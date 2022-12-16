from flask import Blueprint, render_template, request, abort
from jinja2 import TemplateNotFound
from personalwebsite_app.jinja2_env import projects_base_structure_jinja2_env
from torch import no_grad
from functools import lru_cache

##
from .machine_translation.Transformer_implementation import createVocabs
##

from .machine_translation.Transformer_implementation import loadModel
from .machine_translation.Transformer_implementation import translateUserInput
from .machine_translation.Transformer_implementation import standardizeOutput

project_id_2_bp = Blueprint(name='project_id_2_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/project_id_2',
                    url_prefix="/project2");

### -- ENGLISH -> FRENCH IMPLEMENTATION -- ###

## MODEL INSTANTIATION ##
@lru_cache(maxsize=None)
def modelInstantiation():
    en_to_fr_model, en_source_vocab, fr_target_vocab  = loadModel(load_parameters=True, load_on_cpu=True);
    en_to_fr_model.eval();
    return en_to_fr_model, en_source_vocab, fr_target_vocab;

## VIEWS ##
@project_id_2_bp.route('/')
def project2():

    ##
    # createVocabs(True);
    ##

    project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
    return render_template('project2.html', 
                            project_base_html_loader=project_base_html_loader);


@project_id_2_bp.route('/en', methods=['POST'])
def translateEnglishToFrench():
    en_to_fr_model, en_source_vocab, fr_target_vocab = modelInstantiation();

    en_input = request.form['input'].lower();
    with no_grad():
        en_to_fr_translation = translateUserInput(en_input, en_to_fr_model, en_source_vocab, fr_target_vocab);

    project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
    return render_template('project2.html', 
                            project_base_html_loader=project_base_html_loader,
                            en_input = en_input,
                            en_to_fr_translation=en_to_fr_translation);

### -- ENGLISH <-> FRENCH IMPLEMENTATION -- ###

# ## MODEL INSTANTIATION ##

# @cache
# def modelInstantiation(en_to_fr:bool):
#     if en_to_fr:
#         en_to_fr_model, en_source_vocab, fr_target_vocab  = loadModel(en_to_fr=True, load_parameters=True, load_on_cpu=True);
#         en_to_fr_model.eval();
#         return en_to_fr_model, en_source_vocab, fr_target_vocab;
#     else:
#         fr_to_en_model, fr_source_vocab, en_target_vocab  = loadModel(en_to_fr=False, load_parameters=True, load_on_cpu=True);
#         fr_to_en_model.eval();
#         return fr_to_en_model, fr_source_vocab, en_target_vocab;

# ## VIEWS ##
# @project_id_2_bp.route('/')
# def project2():
#
#     project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
#     return render_template('project2.html', 
#                             project_base_html_loader=project_base_html_loader, 
#                             en_to_fr_display="block", 
#                             fr_to_en_display="none");


# @project_id_2_bp.route('/en', methods=['POST'])
# def translateEnglishToFrench():
    
#     en_to_fr_model, en_source_vocab, fr_target_vocab = modelInstantiation(True);

#     en_input = request.form['input'].lower();
#     with no_grad():
#         en_to_fr_translation = translateUserInput(en_input, en_to_fr_model, en_source_vocab, fr_target_vocab);

#     project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
#     return render_template('project2.html', 
#                             project_base_html_loader=project_base_html_loader,
#                             en_input = en_input,
#                             en_to_fr_translation=en_to_fr_translation,
#                             en_to_fr_display="block", 
#                             fr_to_en_display="none");



# @project_id_2_bp.route('/fr', methods=['POST'])
# def translateFrenchToEnglish():
#     fr_to_en_model, fr_source_vocab, en_target_vocab = modelInstantiation(False);

#     fr_input = request.form['input'].lower();
#     with no_grad():
#         fr_to_en_translation = translateUserInput(fr_input, fr_to_en_model, fr_source_vocab, en_target_vocab);

#     project_base_html_loader = projects_base_structure_jinja2_env.get_template("projects_base_structure.html");
#     return render_template('project2.html', 
#                             project_base_html_loader=project_base_html_loader,
#                             fr_input=fr_input,
#                             fr_to_en_translation=fr_to_en_translation,
#                             en_to_fr_display="none", 
#                             fr_to_en_display="block");
