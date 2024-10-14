from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
import spacy
import bcrypt
import uuid
import re
import os
import json
import random
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
# Load the SpaCy model
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(CustomJSONEncoder, self).default(obj)

app.json_encoder = CustomJSONEncoder
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    text = str(text).lower()
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)


connection_url = os.environ['DATABASE']
engine = create_engine(connection_url)

rdf = pd.read_csv('https://raw.githubusercontent.com/Gyaanendra/gfg-hackfest/refs/heads/main/data/postgres_data.csv')

# Replace NaN values with an empty string and translate TranslatedInstructions
rdf['CleanedIngredients'] = rdf['CleanedIngredients'].fillna('')


# Apply TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(rdf['CleanedIngredients'])

def calculate_similarity(user_ingredients, user_prep_time, user_cook_time):
    user_ingredients_text = preprocess_text(', '.join(user_ingredients))
    user_tfidf = tfidf_vectorizer.transform([user_ingredients_text])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)[0]
    
    prep_time_similarity = 1 - abs(rdf['PrepTimeInMins'] - user_prep_time) / rdf['PrepTimeInMins'].max()
    cook_time_similarity = 1 - abs(rdf['CookTimeInMins'] - user_cook_time) / rdf['CookTimeInMins'].max()
    
    min_length = min(len(cosine_similarities), len(prep_time_similarity), len(cook_time_similarity))
    cosine_similarities = cosine_similarities[:min_length]
    prep_time_similarity = prep_time_similarity[:min_length]
    cook_time_similarity = cook_time_similarity[:min_length]
    
    combined_similarity = (cosine_similarities + prep_time_similarity + cook_time_similarity) / 3
    return combined_similarity

def recommend_recipes(user_ingredients, user_prep_time, user_cook_time, top_n=10):
    combined_similarity = calculate_similarity(user_ingredients, user_prep_time, user_cook_time)
    sorted_indices = combined_similarity.argsort()[::-1]
    top_recommendations = rdf.iloc[sorted_indices[:top_n]].copy()
    return top_recommendations

@app.route('/api/display_recipe', methods=['GET'])
def display_recipe():
    try:
        # Randomly sample 50 recipes
        recipes_sample = rdf.sample(n=min(50, len(rdf)), random_state=random.randint(0, 10000))
        
        recipes = recipes_sample[[  # Select the relevant columns
            'TranslatedRecipeName',
            'TranslatedIngredients',
            'PrepTimeInMins',
            'CookTimeInMins',
            'TotalTimeInMins',
            'Servings',
            'Cuisine',
            'Course',
            'Diet',
            'TranslatedInstructions',
            'image_src',
            'unique_id'
        ]].to_dict(orient='records')
        
        # Convert numpy int64 and float64 to Python int and float
        for recipe in recipes:
            for key, value in recipe.items():
                if isinstance(value, np.int64):
                    recipe[key] = int(value)
                elif isinstance(value, np.float64):
                    recipe[key] = float(value)
        
        # Ensure all data is JSON serializable
        json_compatible_recipes = json.loads(json.dumps(recipes, default=str))
        
        return jsonify(json_compatible_recipes)
    except Exception as e:
        app.logger.error(f"Error in display_recipe: {str(e)}")
        return jsonify({"error": "An error occurred while fetching recipes"}), 500

@app.route('/api/recommendation', methods=['POST'])
def recommendation():
    data = request.get_json()

    user_ingredients = data.get("user_ingredients", [])
    user_prep_time = data.get("user_prep_time", 0)
    user_cook_time = data.get("user_cook_time", 0)
    n_recipes = data.get("n_recipes", 10)

    if not user_ingredients or user_prep_time <= 0 or user_cook_time <= 0:
        return jsonify({"error": "Invalid input provided."}), 400

    recommendations = recommend_recipes(user_ingredients, user_prep_time, user_cook_time, top_n=n_recipes)

    response_data = recommendations[[
        'TranslatedRecipeName',
        'TranslatedIngredients',
        'PrepTimeInMins',
        'CookTimeInMins',
        'TotalTimeInMins',
        'Servings',
        'Cuisine',
        'Course',
        'Diet',
        'TranslatedInstructions',
        'image_src',
        'unique_id'
    ]].to_dict(orient='records')

    return jsonify(response_data)


@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()

    # Extract user data
    firstname = data.get('firstname')
    lastname = data.get('lastname')
    email = data.get('email')
    password = data.get('password')

    # Validate input
    if not all([firstname, lastname, email, password]):
        return jsonify({"error": "All fields are required"}), 400

    # Validate email format
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"error": "Invalid email format"}), 400

    # Check if email already exists
    with engine.connect() as connection:
        result = connection.execute(text("SELECT * FROM users WHERE email = :email"), {"email": email})
        if result.fetchone():
            return jsonify({"error": "Email already registered"}), 409

    # Generate UUID
    user_id = str(uuid.uuid4())

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert new user into the database
    try:
        with engine.connect() as connection:
            query = text("""
                INSERT INTO users (id, firstname, lastname, email, password)
                VALUES (:id, :firstname, :lastname, :email, :password)
            """)
            connection.execute(query, {
                "id": user_id,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "password": hashed_password.decode('utf-8')
            })
            connection.commit()

        return jsonify({"message": "User registered successfully", "user_id": user_id}), 201

    except Exception as e:
        print(f"Error during user registration: {e}")
        return jsonify({"error": "An error occurred during registration"}), 500
     
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    # Extract login data
    email = data.get('email')
    password = data.get('password')

    # Validate input
    if not all([email, password]):
        return jsonify({"error": "Email and password are required"}), 400

    # Check if user exists and verify password
    try:
        with engine.connect() as connection:
            query = text("SELECT id, firstname, lastname, email, password FROM users WHERE email = :email")
            result = connection.execute(query, {"email": email}).fetchone()

            if result is None:
                return jsonify({"error": "Invalid email or password"}), 401

            user_id, firstname, lastname, db_email, hashed_password = result

            if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                return jsonify({
                    "message": "Login successful",
                    "user_id": user_id,
                    "firstname": firstname,
                    "lastname": lastname,
                    "email": db_email
                }), 200
            else:
                return jsonify({"error": "Invalid email or password"}), 401

    except Exception as e:
        print(f"Error during login: {e}")
        return jsonify({"error": "An error occurred during login"}), 500

if __name__ == '__main__':
    app.run(debug=True)