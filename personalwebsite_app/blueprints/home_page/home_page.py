from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound
from personalwebsite_app.jinja2_env import base_structure_jinja2_env

home_bp = Blueprint(name='home_page_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/home_page',
                    url_prefix='/home');

@home_bp.route('/')
def home():
    base_html_loader = base_structure_jinja2_env.get_template("base.html");
    return render_template('home.html', base_html_loader=base_html_loader);