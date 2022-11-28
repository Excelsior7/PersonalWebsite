from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

contact_bp = Blueprint(name='contact_page_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/contact_page',
                    url_prefix="/contact");

@contact_bp.route('/')
def contact():
    return render_template('contact.html');