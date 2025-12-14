from flask import Flask
try:
    from authlib.integrations.flask_client import OAuth
    import requests
    print("Dependencies found.")
except ImportError as e:
    print(f"Import Error: {e}")
    exit(1)

import os

app = Flask(__name__)
# Exact values from user
app.config['GITHUB_CLIENT_ID'] = 'Ov23lirz0U8ygpsWMKg8'
app.config['GITHUB_CLIENT_SECRET'] = '7d3a6845aafb376749677d5e716ea023040bddd8'

print("Attempting OAuth Init...")
try:
    oauth = OAuth(app)
    github = oauth.register(
        name='github',
        client_id=app.config['GITHUB_CLIENT_ID'],
        client_secret=app.config['GITHUB_CLIENT_SECRET'],
        access_token_url='https://github.com/login/oauth/access_token',
        access_token_params=None,
        authorize_url='https://github.com/login/oauth/authorize',
        authorize_params=None,
        api_base_url='https://api.github.com/',
        client_kwargs={'scope': 'user:email'},
    )
    print(f"SUCCESS: GitHub Object Created: {github}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
