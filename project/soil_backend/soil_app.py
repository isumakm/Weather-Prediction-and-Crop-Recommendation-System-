from flask import Flask
from flask_cors import CORS
from soil_backend.soil_routes import soil_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(soil_bp)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
