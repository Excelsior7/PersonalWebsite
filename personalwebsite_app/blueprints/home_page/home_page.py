from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

home_bp = Blueprint(name='home_page_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/home_page',
                    url_prefix="/home");
