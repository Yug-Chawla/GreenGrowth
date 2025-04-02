import numpy as np
import pickle
import sys
import logging
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# Configure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Suppress version mismatch warnings
warnings.simplefilter("ignore", InconsistentVersionWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create app
app = Flask(__name__)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load models safely
try:
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("model/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    logging.info("Models loaded successfully!")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier

    scaler = StandardScaler()
    pca = PCA(n_components=3)
    rf_model = RandomForestClassifier()
    label_encoder = LabelEncoder()

    logging.warning("⚠️ Using fallback models (not accurate predictions).")

# Sign-Up Route (Initial Page)
@app.route('/')
def sign_up():
    return render_template('sign.html')

# Home route (Page after Sign Up)
@app.route('/home')
def home():
    return render_template('index.html', show_result=False)

# Sign-Up Form Submit Route
@app.route('/sign_up_submit', methods=['POST'])
def sign_up_submit():
    # Here you would handle saving the user's info into a database or session
    username = request.form.get('username', '')
    password = request.form.get('password', '')  # Secure password handling is advised

    if username and password:  # Check if data is valid
        flash("Sign up successful! Redirecting to the home page.")
        return redirect(url_for('home'))  # Use `url_for` to ensure correct routing to home
    else:
        flash("Please fill in all fields.")
        return redirect(url_for('sign'))  # Stay on sign-up page if something is missing

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = {
            'N': request.form.get('N', ''),
            'P': request.form.get('P', ''),
            'K': request.form.get('K', ''),
            'temperature': request.form.get('temperature', ''),
            'humidity': request.form.get('humidity', ''),
            'ph': request.form.get('ph', ''),
            'rainfall': request.form.get('rainfall', ''),
            'label': request.form.get('label', '')
        }

        logging.info(f"Received form data: {form_data}")

        # Validate input data
        try:
            input_values = [float(form_data[field]) for field in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

            if not (0 <= float(form_data['ph']) <= 14):
                raise ValueError("⚠️ pH must be between 0 and 14")

            if not (0 <= float(form_data['humidity']) <= 100):
                raise ValueError("⚠️ Humidity must be between 0 and 100%")

        except ValueError as ve:
            logging.error(f"Validation error: {ve}")
            return render_template('index.html', error=f"❌ Invalid input: {str(ve)}", show_result=True, form_data=form_data)

        logging.info(f"Validated input data: {input_values}")

        # Preprocess input
        test_input = np.array([input_values])
        test_input_scaled = scaler.transform(test_input)
        test_input_pca = pca.transform(test_input_scaled)

        field_label = form_data['label']
        field_info = f" for {field_label}" if field_label else ""

        # Predict crop
        predicted_label_index = rf_model.predict(test_input_pca)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

        logging.info(f"Predicted Crop: {predicted_label}")

        # Get top 3 alternative crops
        try:
            proba = rf_model.predict_proba(test_input_pca)[0]
            top_indices = proba.argsort()[-4:-1][::-1]
            alternatives = [label_encoder.inverse_transform([idx])[0] for idx in top_indices]
            alternatives_str = ", ".join(alternatives)
        except:
            alternatives_str = "Not available"

        return render_template('index.html',
                               prediction=f"✅ Recommended Crop{field_info}: {predicted_label}",
                               alternatives=f"🔹 Alternative crops: {alternatives_str}",
                               show_result=True,
                               form_data=form_data)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', error=f"⚠️ Error processing request: {str(e)}", show_result=True, form_data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
