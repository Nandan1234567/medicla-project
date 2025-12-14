try:
    import authlib
    print(f"Authlib version: {authlib.__version__}")
    from authlib.integrations.flask_client import OAuth
    print("Authlib.integrations.flask_client imported successfully")
except Exception as e:
    print(f"Import Failed: {e}")
