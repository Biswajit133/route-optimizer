"""Pathoconnect Copilot service - its own process, separate from app.py.

Run locally:   python copilot_app.py            (port 5010)
Production:    gunicorn copilot_app:app --worker-class gthread --threads 8 --timeout 120
               (install with: pip install -r requirements-copilot.txt)

A separate gunicorn process because an LLM answer takes seconds; sharing
app.py's single sync worker would block the route/delay/no-show endpoints.
"""
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

load_dotenv()

logging.basicConfig(level=os.environ.get('COPILOT_LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')

from copilot.routes import bp as copilot_bp  # noqa: E402  (after load_dotenv)

app = Flask(__name__)

CORS_ORIGINS = [o.strip() for o in os.environ.get('CORS_ORIGINS', 'http://localhost:4200').split(',') if o.strip()]
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

app.register_blueprint(copilot_bp)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'copilot'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5010)))
