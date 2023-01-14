from enum import Enum
from flask import Blueprint, render_template, redirect, url_for
from personalwebsite_app.jinja2_env import base_structure_jinja2_env

class PROJECT_TYPE(Enum): 
    CV = "CV";
    DL = "DL";
    GM = "GM";
    IR = "IR";
    ML = "ML";
    NLP = "NLP";
    RS = "RS";
    DS = "DS";
    SWE = "SWE";

class Projects():
    def __init__(self, id: int, project_type: PROJECT_TYPE, project_title: str, project_description: str):
        self.id = id;
        self.project_type = project_type.value;
        self.project_title = project_title;
        self.project_description = project_description;

projects_objects = [
    Projects(
        3,
        PROJECT_TYPE.DS,
        "NBA Dashboard", 
        "Some key statistics up to date on the NBA teams presented in the form of a dashboard."),
    Projects(
        2,
        PROJECT_TYPE.NLP,
        "Machine Translation EN->FR", 
        "Simple Transformer-based architecture machine translation algorithm trained from scratch: English -> French"),
    Projects(
        1,
        PROJECT_TYPE.CV,
        "Object Detection in Video", 
        "Given an input video, produce the same video with bounding boxes around the object of interest to the user."),
    # Projects(
    #     4,
    #     PROJECT_TYPE.IR,
    #     "Information Retrieval", 
    #     "Propose a list of movies ordered by relevance according to the user's query."),
];


projects_page_bp = Blueprint(name='projects_page_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/projects_page',
                    url_prefix="/projects");

@projects_page_bp.route('/')
def projects():
    base_html_loader = base_structure_jinja2_env.get_template("base.html");
    return render_template('projects.html', base_html_loader=base_html_loader, projects_objects=projects_objects);

@projects_page_bp.route('/<int:id>')
def projectsRouting(id):
    id = str(id);
    return redirect(url_for('project_id_'+id+'_bp.project'+id));
