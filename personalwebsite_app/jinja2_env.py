import jinja2
import os

base_structure_jinja2_env =jinja2.Environment(
    loader=jinja2.FileSystemLoader(searchpath=os.path.join(os.path.dirname(__file__),'./blueprints/base_structure/templates/base_structure')),
    autoescape=jinja2.select_autoescape()
);

projects_base_structure_jinja2_env =jinja2.Environment(
    loader=jinja2.FileSystemLoader(searchpath=os.path.join(os.path.dirname(__file__),'./blueprints/projects/projects_page/templates/projects_page')),
    autoescape=jinja2.select_autoescape()
);