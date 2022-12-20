import os
from flask import Flask, url_for, redirect
from werkzeug.exceptions import InternalServerError


def create_app(test_config=None):
    # create and configure the app
    app = Flask("personalwebsite_app", instance_relative_config=True);
    app.config.from_mapping(
        SECRET_KEY='dev'
    );

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True);
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config);

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path);
    except OSError:
        pass;

    @app.route('/', methods=['GET'])
    def root():
        return redirect(url_for('home_page_bp.home'));

    #------------BLUEPRINTS------------#
    from personalwebsite_app.blueprints.base_structure.base_structure import base_bp
    app.register_blueprint(base_bp);
    from personalwebsite_app.blueprints.exceptions.exceptions import exceptions_bp
    app.register_blueprint(exceptions_bp);
    from personalwebsite_app.blueprints.cv_page.cv_page import cv_bp
    app.register_blueprint(cv_bp);
    from personalwebsite_app.blueprints.home_page.home_page import home_bp
    app.register_blueprint(home_bp);
    from personalwebsite_app.blueprints.projects.projects_page.projects_page import projects_page_bp
    app.register_blueprint(projects_page_bp);

    from personalwebsite_app.blueprints.projects.project_id_1.project_id_1 import project_id_1_bp
    app.register_blueprint(project_id_1_bp);
    from personalwebsite_app.blueprints.projects.project_id_2.project_id_2 import project_id_2_bp
    app.register_blueprint(project_id_2_bp);
    from personalwebsite_app.blueprints.projects.project_id_3.project_id_3 import project_id_3_bp
    app.register_blueprint(project_id_3_bp);
    from personalwebsite_app.blueprints.projects.project_id_4.project_id_4 import project_id_4_bp
    app.register_blueprint(project_id_4_bp);


    #------------EXCEPTIONS------------#
    import personalwebsite_app.blueprints.exceptions.exceptions as exceptions
    app.register_error_handler(InternalServerError, exceptions.internalServerError);
    app.register_error_handler(404, exceptions.pageNotFound404);

    return app;

app = create_app();