from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

projects_page_bp = Blueprint(name='projects_page_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/projects_page',
                    url_prefix="/projects");

