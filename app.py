import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
import joblib
import logging
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create the database
with app.app_context():
    db.create_all()

# Load the trained model and vectorizer
try:
    model = joblib.load('fake_review_model.pkl')  # Path to the saved model
    vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Path to the saved vectorizer
    logging.info("Model and vectorizer loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model/vectorizer: {e}")
    raise
@app.route('/robots.txt')
def robots():
    return app.send_static_file('robots.txt')

@app.route('/sitemap.xml')
def sitemap():
    return app.send_static_file('sitemap.xml')
# Define the home route
@app.route('/')
@login_required
def home():
    return render_template('index.html')

# Define the login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

# Define the register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful, please log in')
            return redirect(url_for('login'))
    return render_template('register.html')
    @app.route('/google<verification_code>.html')
def google_verify():
    return app.send_static_file('google<verification_code>.html')
# Define the logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Define the about route
@app.route('/about')
def about():
    return render_template('about.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get the review text from the form
        review_text = request.form['review']
        
        # Handle empty review input
        if not review_text.strip():
            return render_template('index.html', error="Error: Please enter a review.", prediction=None, explanation=None)
        
        # Transform the review text using the saved vectorizer
        review_tfidf = vectorizer.transform([review_text])
        
        # Predict using the loaded model
        prediction = model.predict(review_tfidf)
        result = "Fake" if prediction[0] == 1 else "Genuine"
        
        # Determine the emoji to display
        emoji = "😊" if result == "Genuine" else "😢"
        
        # Return the result and emoji to the user
        return render_template('index.html', prediction=result, emoji=emoji)
    
    except Exception as e:
        logging.error(f"Error processing review: {e}")
        return render_template('index.html', error="Error: Unable to process the review.", prediction=None, emoji=None)

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

# Step 1: Load the dataset
def load_review_data(file_path):
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        
        # Keep only relevant columns: 'Text' (review text) and 'Label' (fake/genuine)
        data = data[['Text', 'Label']]
        
        # Drop rows with missing values
        data.dropna(inplace=True)
        
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)


file_path = 'Amazon_Reviews.csv'
data = load_review_data(file_path)

X = data['Text']
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Data split into training and testing sets.")

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
logging.info("Feature extraction completed using TF-IDF.")


param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
"""
#start_time = time.time()
#grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1, verbose=2)  # Reduced number of CV folds
#grid_search.fit(X_train_tfidf, y_train)
#end_time = time.time()

#model = grid_search.best_estimator_
#logging.info(f"Best parameters found: {grid_search.best_params_}")
#logging.info("Model training completed.")
#logging.info(f"Training time: {end_time - start_time} seconds")

#y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
logging.info(f"Model evaluation completed. Accuracy: {accuracy}")
logging.info(f"\nClassification Report:\n{classification_rep}")
joblib.dump(model, 'fake_review_model.pkl')
#joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
#logging.info("Model and vectorizer saved successfully.")
*/"""
