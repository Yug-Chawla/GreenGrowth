import numpy as np
import pickle
import sys
import logging
import os
import datetime
import requests
import json
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.exceptions import InconsistentVersionWarning
import warnings
from geopy.geocoders import Nominatim
from functools import wraps
from datetime import datetime, timedelta
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore", InconsistentVersionWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///advanced_agri_ai.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys - Load from environment variables
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')
    SOIL_GRIDS_API_URL = "https://rest.soilgrids.org/soilgrids/v2.0/properties/query"
    
    # File upload configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Enhanced crop database with comprehensive data
COMPREHENSIVE_CROP_DATA = {
    "rice": {
        "type": "Cereal",
        "hindi_name": "चावल",
        "optimal_ph": [5.5, 6.5],
        "optimal_temp": [20, 35],
        "optimal_rainfall": [1000, 2000],
        "growing_season": "Kharif",
        "duration_days": 120,
        "water_requirement": "high",
        "nitrogen_req": [80, 120],
        "phosphorus_req": [40, 60],
        "potassium_req": [40, 60],
        "base_yield": 3500,
        "base_price": 28,
        "sustainability": 6.5,
        "diseases": ["blast", "sheath_blight", "brown_spot"],
        "pests": ["stem_borer", "leaf_folder"],
        "market_demand": "very_high"
    },
    "wheat": {
        "type": "Cereal",
        "hindi_name": "गेहूं",
        "optimal_ph": [6.0, 7.5],
        "optimal_temp": [15, 25],
        "optimal_rainfall": [300, 600],
        "growing_season": "Rabi",
        "duration_days": 120,
        "water_requirement": "medium",
        "nitrogen_req": [80, 120],
        "phosphorus_req": [40, 60],
        "potassium_req": [40, 60],
        "base_yield": 3000,
        "base_price": 25,
        "sustainability": 7.0,
        "diseases": ["rust", "blight"],
        "pests": ["aphids", "termites"],
        "market_demand": "very_high"
    },
    "maize": {
        "type": "Cereal",
        "hindi_name": "मक्का",
        "optimal_ph": [5.8, 7.2],
        "optimal_temp": [18, 32],
        "optimal_rainfall": [500, 1200],
        "growing_season": "Kharif",
        "duration_days": 100,
        "water_requirement": "medium",
        "nitrogen_req": [60, 100],
        "phosphorus_req": [30, 50],
        "potassium_req": [30, 50],
        "base_yield": 2500,
        "base_price": 22,
        "sustainability": 6,
        "diseases": ["maize_streak", "rust"],
        "pests": ["borer", "armyworm"],
        "market_demand": "high"
    },
    "cotton": {
        "type": "Staple Crop",
        "hindi_name": "कपास",
        "optimal_ph": [5.5, 7.5],
        "optimal_temp": [25, 35],
        "optimal_rainfall": [600, 1200],
        "growing_season": "Kharif",
        "duration_days": 150,
        "water_requirement": "medium",
        "nitrogen_req": [50, 100],
        "phosphorus_req": [30, 50],
        "potassium_req": [40, 60],
        "base_yield": 1500,
        "base_price": 70,
        "sustainability": 6,
        "diseases": ["bacterial_blight", "boll_rot"],
        "pests": ["bollworm", "aphids"],
        "market_demand": "high"
    },
    "chickpea": {
        "type": "Pulse",
        "hindi_name": "चना",
        "optimal_ph": [6.0, 7.5],
        "optimal_temp": [15, 30],
        "optimal_rainfall": [300, 600],
        "growing_season": "Rabi",
        "duration_days": 110,
        "water_requirement": "low",
        "nitrogen_req": [20, 40],
        "phosphorus_req": [40, 60],
        "potassium_req": [30, 50],
        "base_yield": 1200,
        "base_price": 75,
        "sustainability": 8.5,
        "diseases": ["blight", "wilt"],
        "pests": ["pod_borer", "aphids"],
        "market_demand": "very_high"
    },
    "soybean": {
        "type": "Oilseed",
        "hindi_name": "सोयाबीन",
        "optimal_ph": [6.0, 7.0],
        "optimal_temp": [20, 30],
        "optimal_rainfall": [500, 1000],
        "growing_season": "Kharif",
        "duration_days": 95,
        "water_requirement": "medium",
        "nitrogen_req": [25, 40],
        "phosphorus_req": [60, 80],
        "potassium_req": [40, 60],
        "base_yield": 1800,
        "base_price": 55,
        "sustainability": 8.0,
        "diseases": ["rust", "blight"],
        "pests": ["stem_borer", "aphids"],
        "market_demand": "high"
    }
}

app = Flask(__name__)
app.config.from_object(Config)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("models", exist_ok=True)

db = SQLAlchemy(app)

# Initialize the geolocator
try:
    geolocator = Nominatim(user_agent="advanced_agri_ai_app_v1", timeout=10)
except Exception as e:
    logging.warning(f"Geolocator initialization failed: {e}")
    geolocator = None

# Security decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(15))
    password_hash = db.Column(db.String(128), nullable=False)
    preferred_language = db.Column(db.String(5), default='en')
    location_lat = db.Column(db.Float)
    location_lng = db.Column(db.Float)
    farm_size_hectares = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class CropRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Location and environmental data
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    
    # Soil parameters
    soil_ph = db.Column(db.Float, nullable=False)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    
    # Climate data
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    
    # Recommendations and predictions
    recommended_crop = db.Column(db.String(100), nullable=False)
    alternative_crops = db.Column(db.String(500))
    confidence_score = db.Column(db.Float)
    
    # Economic predictions
    estimated_yield = db.Column(db.Float)
    estimated_revenue = db.Column(db.Float)
    estimated_cost = db.Column(db.Float)
    estimated_profit = db.Column(db.Float)
    roi_percentage = db.Column(db.Float)

# Utility Functions
def get_location_from_coordinates(lat, lon):
    """Get location details from coordinates"""
    try:
        if geolocator:
            location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
            
            if location and location.raw.get('address'):
                address = location.raw['address']
                return {
                    'city': address.get('city', address.get('town', address.get('village', 'Unknown'))),
                    'state': address.get('state', 'Unknown'),
                    'country': address.get('country', 'Unknown'),
                    'full_address': location.address
                }
        
        return {'city': 'Unknown', 'state': 'Unknown', 'country': 'Unknown', 'full_address': f'Lat: {lat}, Lon: {lon}'}
            
    except Exception as e:
        logging.error(f"Reverse geocoding error for ({lat}, {lon}): {e}")
        return {'city': 'Unknown', 'state': 'Unknown', 'country': 'Unknown', 'full_address': f'Lat: {lat}, Lon: {lon}'}

# Weather Service
class WeatherService:
    @staticmethod
    def get_current_weather(lat, lon):
        """Get current weather from OpenWeatherMap or return mock data"""
        if not app.config.get('OPENWEATHER_API_KEY'):
            logging.info("OpenWeatherMap API key not configured, using mock data")
            return WeatherService._get_mock_weather_data(lat, lon)
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': app.config['OPENWEATHER_API_KEY'],
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'description': data['weather'][0]['description'],
                'rainfall': data.get('rain', {}).get('1h', 0) * 24  # Convert to daily
            }
        except Exception as e:
            logging.error(f"Weather API error: {e}")
            return WeatherService._get_mock_weather_data(lat, lon)
    
    @staticmethod
    def _get_mock_weather_data(lat, lon):
        """Provide location-based mock weather data"""
        import random
        
        # Adjust based on latitude (rough climate zones)
        if lat > 30:  # Northern regions
            temp_base = 22
            rainfall_base = 600
        elif lat > 25:  # Northern plains
            temp_base = 28
            rainfall_base = 800
        elif lat > 15:  # Central/Western India
            temp_base = 32
            rainfall_base = 700
        else:  # Southern India
            temp_base = 30
            rainfall_base = 1200
        
        return {
            'temperature': temp_base + random.uniform(-5, 8),
            'humidity': random.uniform(60, 85),
            'pressure': random.uniform(1010, 1025),
            'wind_speed': random.uniform(2, 10),
            'description': 'Partly cloudy',
            'rainfall': rainfall_base + random.uniform(-200, 400)
        }

# Soil Service
class SoilService:
    @staticmethod
    def get_soil_data_from_soilgrids(lat, lon):
        """Get soil data from SoilGrids API"""
        try:
            url = app.config['SOIL_GRIDS_API_URL']
            params = {
                'lon': lon,
                'lat': lat,
                'property': ['phh2o', 'nitrogen', 'cec'],
                'depth': ['0-5cm'],
                'value': ['mean']
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            soil_props = {}
            
            for prop in data.get('properties', []):
                prop_name = prop['name']
                depths = prop.get('depths', [])
                if depths and depths[0].get('values'):
                    surface_value = depths[0]['values']['mean']
                    
                    if prop_name == 'phh2o':
                        soil_props['ph'] = surface_value / 10
                    elif prop_name == 'nitrogen':
                        soil_props['nitrogen'] = surface_value / 100
                    else:
                        soil_props[prop_name] = surface_value
            
            return soil_props
            
        except Exception as e:
            logging.error(f"SoilGrids API error: {e}")
            return None
    
    @staticmethod
    def estimate_soil_properties(lat, lon):
        """Estimate soil properties using SoilGrids or regional defaults"""
        # Try to get data from SoilGrids first
        soil_data = SoilService.get_soil_data_from_soilgrids(lat, lon)
        
        if not soil_data:
            soil_data = {}
        
        # Fill missing values with regional defaults
        defaults = SoilService._get_regional_soil_defaults(lat, lon)
        for key, value in defaults.items():
            if key not in soil_data or soil_data[key] is None:
                soil_data[key] = value
        
        return soil_data
    
    @staticmethod
    def _get_regional_soil_defaults(lat, lon):
        """Regional soil defaults for different regions of India"""
        if not (8 <= lat <= 37 and 68 <= lon <= 97):
            return {'ph': 6.8, 'nitrogen': 65, 'phosphorus': 35, 'potassium': 45}
        
        if lat > 30:  # Kashmir region
            return {'ph': 7.5, 'nitrogen': 80, 'phosphorus': 40, 'potassium': 55}
        elif lat > 28:  # Northern plains
            return {'ph': 7.2, 'nitrogen': 75, 'phosphorus': 35, 'potassium': 45}
        elif lat > 23:  # Central India
            return {'ph': 7.0, 'nitrogen': 70, 'phosphorus': 38, 'potassium': 48}
        elif lat > 15:  # Deccan plateau
            return {'ph': 6.8, 'nitrogen': 65, 'phosphorus': 42, 'potassium': 52}
        else:  # Southern India
            return {'ph': 6.5, 'nitrogen': 60, 'phosphorus': 40, 'potassium': 50}

# ML-based Crop Recommendation Engine
class CropRecommendationEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.label_encoder = None
        self.model_loaded = False
        self.load_models()

    def load_models(self):
        """Load pre-trained ML models"""
        model_files = {
            'model': 'model/rf_model.pkl',
            'scaler': 'model/scaler.pkl',
            'label_encoder': 'model/label_encoder.pkl'
            
        }
        
        try:
            missing_files = [path for path in model_files.values() if not os.path.exists(path)]
            
            if missing_files:
                logging.warning(f"Model files not found: {missing_files}. Using fallback recommendations.")
                self.model_loaded = False
                return
            
            with open(model_files['model'], 'rb') as f:
                self.model = pickle.load(f)
            with open(model_files['scaler'], 'rb') as f:
                self.scaler = pickle.load(f)
            with open(model_files['label_encoder'], 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Try to load PCA if exists (optional)
            try:
                with open('models/pca.pkl', 'rb') as f:
                    self.pca = pickle.load(f)
            except:
                self.pca = None
            
            self.model_loaded = True
            logging.info("✅ ML Models loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Model loading failed: {e}")
            self.model_loaded = False

    def recommend_crop(self, soil_data, weather_data, location_data):
        """Get crop recommendations using ML model or fallback"""
        try:
            if self.model_loaded and self.model is not None:
                return self._predict_crops_ml(soil_data, weather_data, location_data)
            else:
                return self._get_fallback_recommendation(soil_data, weather_data, location_data)
        except Exception as e:
            logging.error(f"Crop recommendation error: {e}")
            return self._get_fallback_recommendation(soil_data, weather_data, location_data)
    
    def _predict_crops_ml(self, soil_data, weather_data, location_data):
        """ML-based crop prediction"""
        try:
            # Prepare features in the order your model expects
            # Adjust this based on your model's feature order
            features = np.array([[
                float(soil_data.get('nitrogen', 50)),
                float(soil_data.get('phosphorus', 30)),
                float(soil_data.get('potassium', 40)),
                float(weather_data.get('temperature', 25)),
                float(weather_data.get('humidity', 70)),
                float(soil_data.get('ph', 6.5)),
                float(weather_data.get('rainfall', 800))
            ]])

            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Apply PCA if available
            if self.pca is not None:
                scaled_features = self.pca.transform(scaled_features)
            
            # Get predictions
            prediction = self.model.predict(scaled_features)
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            # Get top 3 recommendations
            top_indices = np.argsort(probabilities)[::-1][:3]
            recommendations = []
            
            for i, idx in enumerate(top_indices):
                crop_name = self.label_encoder.inverse_transform([idx])[0]
                confidence = float(probabilities[idx]) * 100
                
                if confidence > 5:  # Only include if confidence > 5%
                    economics = self._calculate_economics(crop_name, features[0], location_data)
                    sustainability = self._calculate_sustainability(crop_name)
                    
                    rec = {
                        'crop': crop_name,
                        'confidence': round(confidence, 2),
                        'rank': i + 1,
                        'prediction_method': 'ML Model'
                    }
                    rec.update(economics)
                    rec.update(sustainability)
                    recommendations.append(rec)

            return recommendations if recommendations else self._get_fallback_recommendation(soil_data, weather_data, location_data)
            
        except Exception as e:
            logging.error(f"ML prediction error: {e}")
            return self._get_fallback_recommendation(soil_data, weather_data, location_data)
    
    def _get_fallback_recommendation(self, soil_data, weather_data, location_data):
        """Fallback recommendation when ML model fails"""
        lat = location_data.get('lat', 20)
        
        # Basic regional recommendations
        if lat > 28:  # Northern India
            default_crops = ['wheat', 'rice', 'maize']
        elif lat > 20:  # Central India
            default_crops = ['soybean', 'cotton', 'maize']
        else:  # Southern India
            default_crops = ['rice', 'cotton', 'chickpea']
        
        recommendations = []
        for i, crop in enumerate(default_crops):
            crop_info = COMPREHENSIVE_CROP_DATA.get(crop, {})
            rec = {
                'crop': crop,
                'confidence': 85 - (i * 15),
                'rank': i + 1,
                'prediction_method': 'Regional Fallback',
                'estimated_yield': crop_info.get('base_yield', 2000),
                'estimated_revenue': crop_info.get('base_yield', 2000) * crop_info.get('base_price', 25),
                'sustainability_score': crop_info.get('sustainability', 7.0)
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _calculate_economics(self, crop_name, features, location_data):
        """Calculate economic predictions for the crop"""
        crop_info = COMPREHENSIVE_CROP_DATA.get(crop_name, {})
        
        base_yield = crop_info.get('base_yield', 1000)
        base_price = crop_info.get('base_price', 30)
        
        # Adjust yield based on conditions (simplified)
        yield_factor = self._calculate_yield_factor(features, crop_info)
        estimated_yield = base_yield * yield_factor
        
        # Calculate economics
        estimated_revenue = estimated_yield * base_price
        estimated_cost = estimated_revenue * 0.6  # Assume 60% cost ratio
        estimated_profit = estimated_revenue - estimated_cost
        roi = (estimated_profit / estimated_cost) * 100 if estimated_cost > 0 else 0
        
        return {
            'estimated_yield': round(estimated_yield, 2),
            'estimated_revenue': round(estimated_revenue, 2),
            'estimated_cost': round(estimated_cost, 2),
            'estimated_profit': round(estimated_profit, 2),
            'roi_percentage': round(roi, 2)
        }
    
    def _calculate_yield_factor(self, features, crop_info):
        """Calculate yield adjustment factor based on conditions"""
        # Extract features (adjust indices based on your feature order)
        nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall = features
        
        factor = 1.0
        
        # pH optimization
        ph_optimal = crop_info.get('optimal_ph', [6, 7])
        if ph_optimal[0] <= ph <= ph_optimal[1]:
            factor *= 1.1
        else:
            ph_penalty = min(abs(ph - ph_optimal[0]), abs(ph - ph_optimal[1])) / 2
            factor *= max(0.7, 1 - ph_penalty * 0.1)
        
        # Temperature optimization
        temp_optimal = crop_info.get('optimal_temp', [20, 30])
        if temp_optimal[0] <= temperature <= temp_optimal[1]:
            factor *= 1.05
        
        # Nutrient adequacy
        n_req = crop_info.get('nitrogen_req', [50])[0]
        if nitrogen >= n_req:
            factor *= 1.05
        
        return max(0.5, min(1.5, factor))
    
    def _calculate_sustainability(self, crop_name):
        """Calculate sustainability metrics"""
        crop_info = COMPREHENSIVE_CROP_DATA.get(crop_name, {})
        
        base_sustainability = crop_info.get('sustainability', 7.0)
        water_req = crop_info.get('water_requirement', 'medium')
        
        # Water efficiency scoring
        water_scores = {'very_low': 10, 'low': 8.5, 'medium': 7, 'high': 5.5, 'very_high': 4}
        water_efficiency = water_scores.get(water_req, 7)
        
        return {
            'sustainability_score': round(base_sustainability, 2),
            'water_efficiency': round(water_efficiency, 2)
        }

# Initialize the recommendation engine
recommendation_engine = CropRecommendationEngine()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['username', 'email', 'password']
            for field in required_fields:
                if field not in data or not data[field].strip():
                    return jsonify({'success': False, 'message': f'Missing required field: {field}'}), 400
            
            # Validate email format
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            if not re.match(email_pattern, data['email']):
                return jsonify({'success': False, 'message': 'Invalid email format'}), 400
            
            # Check if user exists
            if User.query.filter_by(email=data['email']).first():
                return jsonify({'success': False, 'message': 'Email already registered'}), 400
            
            if User.query.filter_by(username=data['username']).first():
                return jsonify({'success': False, 'message': 'Username already taken'}), 400
            
            # Create new user
            user = User(
                username=data['username'].strip(),
                email=data['email'].strip().lower(),
                phone=data.get('phone', '').strip() or None,
                preferred_language=data.get('language', 'en'),
                farm_size_hectares=float(data['farm_size']) if data.get('farm_size') else None
            )
            user.set_password(data['password'])
            
            db.session.add(user)
            db.session.commit()
            
            session['user_id'] = user.id
            session['language'] = user.preferred_language
            
            logging.info(f"New user registered: {user.username}")
            return jsonify({'success': True, 'message': 'Registration successful'})
            
        except Exception as e:
            logging.error(f"Registration error: {e}")
            db.session.rollback()
            return jsonify({'success': False, 'message': 'Registration failed'}), 500
    
    return render_template('auth.html', mode='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            if not data.get('email') or not data.get('password'):
                return jsonify({'success': False, 'message': 'Email and password required'}), 400
            
            user = User.query.filter_by(email=data['email'].strip().lower()).first()
            
            if user and user.check_password(data['password']) and user.is_active:
                session['user_id'] = user.id
                session['language'] = user.preferred_language
                
                user.last_login = datetime.utcnow()
                db.session.commit()
                
                logging.info(f"User logged in: {user.username}")
                return jsonify({'success': True, 'message': 'Login successful'})
            else:
                return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
                
        except Exception as e:
            logging.error(f"Login error: {e}")
            return jsonify({'success': False, 'message': 'Login failed'}), 500
    
    return render_template('auth.html', mode='login')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get_or_404(session['user_id'])
    recent_recommendations = CropRecommendation.query.filter_by(
        user_id=user.id
    ).order_by(CropRecommendation.date.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                         user=user, 
                         recommendations=recent_recommendations)

# Main API endpoint for crop recommendation
@app.route('/api/recommend-crop', methods=['POST'])
@login_required
def api_recommend_crop():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract coordinates
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        # Get location information from coordinates
        location_data = get_location_from_coordinates(lat, lon)
        location_data.update({'lat': lat, 'lon': lon})
        
        # Auto-fill soil data from coordinates if not provided
        soil_data = data.get('soil', {})
        if not soil_data or not all(k in soil_data for k in ['ph', 'nitrogen', 'phosphorus', 'potassium']):
            auto_soil_data = SoilService.estimate_soil_properties(lat, lon)
            # Use provided values or auto-filled values
            soil_data = {
                'ph': soil_data.get('ph') or auto_soil_data.get('ph', 6.5),
                'nitrogen': soil_data.get('nitrogen') or auto_soil_data.get('nitrogen', 50),
                'phosphorus': soil_data.get('phosphorus') or auto_soil_data.get('phosphorus', 30),
                'potassium': soil_data.get('potassium') or auto_soil_data.get('potassium', 40)
            }
            logging.info(f"Auto-filled soil data for ({lat}, {lon}): {soil_data}")
        
        # Auto-fill weather data from coordinates if not provided
        weather_data = data.get('weather', {})
        if not weather_data or not all(k in weather_data for k in ['temperature', 'humidity', 'rainfall']):
            auto_weather_data = WeatherService.get_current_weather(lat, lon)
            # Use provided values or auto-filled values
            weather_data = {
                'temperature': weather_data.get('temperature') or auto_weather_data.get('temperature', 25),
                'humidity': weather_data.get('humidity') or auto_weather_data.get('humidity', 70),
                'rainfall': weather_data.get('rainfall') or auto_weather_data.get('rainfall', 800),
                'pressure': auto_weather_data.get('pressure', 1013),
                'wind_speed': auto_weather_data.get('wind_speed', 5),
                'description': auto_weather_data.get('description', 'Clear')
            }
            logging.info(f"Auto-filled weather data for ({lat}, {lon}): {weather_data}")
        
        # Validate data ranges
        if not (0 <= soil_data['ph'] <= 14):
            return jsonify({'error': 'Invalid pH value (must be 0-14)'}), 400
        if not (0 <= weather_data['humidity'] <= 100):
            return jsonify({'error': 'Invalid humidity value (must be 0-100)'}), 400
        
        # Get crop recommendations using ML model
        recommendations = recommendation_engine.recommend_crop(
            soil_data, weather_data, location_data
        )
        
        if not recommendations:
            return jsonify({'error': 'Unable to generate recommendations'}), 500
        
        # Save primary recommendation to database
        primary_rec = recommendations[0]
        
        rec = CropRecommendation(
            user_id=session['user_id'],
            latitude=lat,
            longitude=lon,
            city=location_data.get('city', 'Unknown'),
            state=location_data.get('state', 'Unknown'),
            soil_ph=soil_data['ph'],
            nitrogen=soil_data['nitrogen'],
            phosphorus=soil_data['phosphorus'],
            potassium=soil_data['potassium'],
            temperature=weather_data['temperature'],
            humidity=weather_data['humidity'],
            rainfall=weather_data['rainfall'],
            recommended_crop=primary_rec['crop'],
            alternative_crops=', '.join([r['crop'] for r in recommendations[1:4]]),
            confidence_score=primary_rec.get('confidence', 80),
            estimated_yield=primary_rec.get('estimated_yield'),
            estimated_revenue=primary_rec.get('estimated_revenue'),
            estimated_cost=primary_rec.get('estimated_cost'),
            estimated_profit=primary_rec.get('estimated_profit'),
            roi_percentage=primary_rec.get('roi_percentage')
        )
        
        db.session.add(rec)
        db.session.commit()
        
        logging.info(f"Crop recommendation generated for user {session['user_id']}: {primary_rec['crop']} (Method: {primary_rec.get('prediction_method', 'Unknown')})")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'location': location_data,
            'weather': weather_data,
            'soil': soil_data,
            'recommendation_id': rec.id,
            'auto_filled': {
                'soil': not data.get('soil'),
                'weather': not data.get('weather')
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        logging.error(f"Crop recommendation API error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to generate recommendations'}), 500

# API endpoint to auto-fill data from coordinates
@app.route('/api/auto-fill/<float:lat>/<float:lon>')
@login_required
def api_auto_fill(lat, lon):
    try:
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        # Get all data for the coordinates
        location_data = get_location_from_coordinates(lat, lon)
        soil_data = SoilService.estimate_soil_properties(lat, lon)
        weather_data = WeatherService.get_current_weather(lat, lon)
        
        return jsonify({
            'success': True,
            'location': location_data,
            'soil': soil_data,
            'weather': weather_data
        })
        
    except Exception as e:
        logging.error(f"Auto-fill API error: {e}")
        return jsonify({'error': 'Failed to fetch auto-fill data'}), 500

# Individual data endpoints
@app.route('/api/weather/<float:lat>/<float:lon>')
def api_weather(lat, lon):
    try:
        weather = WeatherService.get_current_weather(lat, lon)
        return jsonify({'success': True, 'weather': weather})
    except Exception as e:
        logging.error(f"Weather API error: {e}")
        return jsonify({'error': 'Failed to fetch weather data'}), 500

@app.route('/api/soil/<float:lat>/<float:lon>')
def api_soil(lat, lon):
    try:
        soil = SoilService.estimate_soil_properties(lat, lon)
        return jsonify({'success': True, 'soil': soil})
    except Exception as e:
        logging.error(f"Soil API error: {e}")
        return jsonify({'error': 'Failed to fetch soil data'}), 500

@app.route('/api/location/<float:lat>/<float:lon>')
def api_location(lat, lon):
    try:
        location = get_location_from_coordinates(lat, lon)
        return jsonify({'success': True, 'location': location})
    except Exception as e:
        logging.error(f"Location API error: {e}")
        return jsonify({'error': 'Failed to fetch location data'}), 500

@app.route('/api/crop-info/<crop_name>')
def api_crop_info(crop_name):
    try:
        crop_info = COMPREHENSIVE_CROP_DATA.get(crop_name.lower())
        if not crop_info:
            return jsonify({'error': 'Crop not found'}), 404
        return jsonify({'success': True, 'crop_info': crop_info})
    except Exception as e:
        logging.error(f"Crop info API error: {e}")
        return jsonify({'error': 'Failed to fetch crop information'}), 500

# Get user's recommendation history
@app.route('/api/recommendations')
@login_required
def api_recommendations():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        recommendations = CropRecommendation.query.filter_by(
            user_id=session['user_id']
        ).order_by(CropRecommendation.date.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'success': True,
            'recommendations': [{
                'id': rec.id,
                'date': rec.date.isoformat(),
                'crop': rec.recommended_crop,
                'alternatives': rec.alternative_crops,
                'confidence': rec.confidence_score,
                'location': f"{rec.city}, {rec.state}",
                'yield': rec.estimated_yield,
                'revenue': rec.estimated_revenue,
                'profit': rec.estimated_profit,
                'roi': rec.roi_percentage
            } for rec in recommendations.items],
            'pagination': {
                'page': recommendations.page,
                'pages': recommendations.pages,
                'per_page': recommendations.per_page,
                'total': recommendations.total
            }
        })
        
    except Exception as e:
        logging.error(f"Recommendations API error: {e}")
        return jsonify({'error': 'Failed to fetch recommendations'}), 500

# Get specific recommendation details
@app.route('/api/recommendation/<int:rec_id>')
@login_required
def api_recommendation_detail(rec_id):
    try:
        rec = CropRecommendation.query.filter_by(
            id=rec_id, user_id=session['user_id']
        ).first_or_404()
        
        return jsonify({
            'success': True,
            'recommendation': {
                'id': rec.id,
                'date': rec.date.isoformat(),
                'location': {
                    'latitude': rec.latitude,
                    'longitude': rec.longitude,
                    'city': rec.city,
                    'state': rec.state
                },
                'soil': {
                    'ph': rec.soil_ph,
                    'nitrogen': rec.nitrogen,
                    'phosphorus': rec.phosphorus,
                    'potassium': rec.potassium
                },
                'weather': {
                    'temperature': rec.temperature,
                    'humidity': rec.humidity,
                    'rainfall': rec.rainfall
                },
                'prediction': {
                    'crop': rec.recommended_crop,
                    'alternatives': rec.alternative_crops,
                    'confidence': rec.confidence_score
                },
                'economics': {
                    'yield': rec.estimated_yield,
                    'revenue': rec.estimated_revenue,
                    'cost': rec.estimated_cost,
                    'profit': rec.estimated_profit,
                    'roi': rec.roi_percentage
                }
            }
        })
        
    except Exception as e:
        logging.error(f"Recommendation detail API error: {e}")
        return jsonify({'error': 'Failed to fetch recommendation details'}), 500

# Authentication status check
@app.route('/api/auth-status')
def auth_status():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            return jsonify({
                'authenticated': True, 
                'username': user.username,
                'user_id': user.id,
                'language': user.preferred_language
            })
    return jsonify({'authenticated': False})

# Model status endpoint
@app.route('/api/model-status')
def model_status():
    return jsonify({
        'model_loaded': recommendation_engine.model_loaded,
        'model_type': 'ML Model' if recommendation_engine.model_loaded else 'Fallback System',
        'available_crops': list(COMPREHENSIVE_CROP_DATA.keys())
    })

# Health check endpoint
@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'services': {
            'database': 'connected',
            'ml_model': 'loaded' if recommendation_engine.model_loaded else 'fallback',
            'weather_api': 'configured' if app.config.get('OPENWEATHER_API_KEY') else 'mock_data',
            'geocoding': 'available' if geolocator else 'unavailable'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            logging.info("✅ Database tables created successfully.")
        except Exception as e:
            logging.error(f"❌ Database initialization failed: {e}")
    
    # Print model loading status
    if recommendation_engine.model_loaded:
        logging.info("🤖 ML models are ready for predictions")
    else:
        logging.warning("⚠️ ML models not found - using fallback recommendations")
    
    logging.info("🌱 Agricultural AI App starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)