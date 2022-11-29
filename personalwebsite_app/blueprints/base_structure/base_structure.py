from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

base_bp = Blueprint(name='base_structure_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/base_structure',
                    url_prefix="/base");



