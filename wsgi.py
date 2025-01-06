from app import app  # Import the Flask application object named 'app'

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)  # You can adjust host and port as needed