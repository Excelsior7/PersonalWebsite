from flask import Blueprint

base_bp = Blueprint(name='base_structure_bp', 
                    import_name=__name__,
                    static_folder='static',
                    template_folder='templates/base_structure',
                    url_prefix="/base");



