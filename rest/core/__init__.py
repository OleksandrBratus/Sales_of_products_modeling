from flask import Flask


def create_app():
    """Construct the core application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config['JSON_SORT_KEYS'] = False

    with app.app_context():
        from . import routes
        return app
