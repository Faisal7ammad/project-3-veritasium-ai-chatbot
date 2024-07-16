from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

def create_app():
    app = Flask(__name__)

    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app