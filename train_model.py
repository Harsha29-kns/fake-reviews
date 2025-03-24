import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Path to the dataset
file_path = 'Amazon_Reviews.csv'  # Place the dataset in the same folder as this script
data = load_review_data(file_path)

# Use a smaller subset of the data for quicker experimentation
# data = data.sample(frac=0.1, random_state=42)

# Step 2: Preprocess and split the data
X = data['Text']
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Data split into training and testing sets.")

# Step 3: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
logging.info("Feature extraction completed using TF-IDF.")

# Step 4: Train a Random Forest model with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],  # Reduced number of estimators
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=2, n_jobs=-1, verbose=2)  # Reduced number of CV folds
grid_search.fit(X_train_tfidf, y_train)

model = grid_search.best_estimator_
logging.info(f"Best parameters found: {grid_search.best_params_}")
logging.info("Model training completed.")

# Step 5: Evaluate the model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
logging.info(f"Model evaluation completed. Accuracy: {accuracy}")
logging.info(f"\nClassification Report:\n{classification_rep}")

# Step 6: Save the model and vectorizer
joblib.dump(model, 'fake_review_model.pkl')  # Save the trained model
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save the vectorizer
logging.info("Model and vectorizer saved successfully.")