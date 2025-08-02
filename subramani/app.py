from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- 2. Load All Model Assets ---
print("Loading all model assets...")
try:
    # K-NN assets
    with open('knn_model.pkl', 'rb') as file:
        model_knn = pickle.load(file)
    with open('utility_matrix.pkl', 'rb') as file:
        utility_matrix = pickle.load(file)
    
    # XGBoost model
    with open('xgboost_model.pkl', 'rb') as file:
        model_xgb = pickle.load(file)

    # Logistic Regression model
    with open('logistic_regression_model.pkl', 'rb') as file:
        model_lr = pickle.load(file)

    # --- FIX 1: Recreate the XGBoost preprocessor ---
    print("Recreating XGBoost preprocessor from raw data...")
    df_xgb_raw = pd.read_csv('student_performance_data.csv')
    X_xgb_raw = df_xgb_raw.drop(['student_id', 'final_outcome'], axis=1)
    numerical_features_xgb = X_xgb_raw.select_dtypes(include=np.number).columns
    categorical_features_xgb = X_xgb_raw.select_dtypes(include=['object']).columns
    preprocessor_xgb = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_xgb),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_xgb)
        ])
    preprocessor_xgb.fit(X_xgb_raw)
    
    # --- FIX 2: Recreate the Dropout preprocessor ---
    print("Recreating Dropout preprocessor from raw data...")
    df_dropout_raw = pd.read_csv('new_dropout_data.csv')
    X_dropout_raw = df_dropout_raw.drop(['student_id', 'dropout_status'], axis=1)
    numerical_features_lr = X_dropout_raw.select_dtypes(include=np.number).columns
    categorical_features_lr = X_dropout_raw.select_dtypes(include=['object', 'bool']).columns
    preprocessor_lr = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_lr),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_lr)
        ])
    preprocessor_lr.fit(X_dropout_raw)

    print("All assets loaded and preprocessors are ready!")

except FileNotFoundError as e:
    print(f"Error loading assets: {e}")
    print("Please ensure all .pkl and required .csv files are in the same folder.")
    exit()

# --- Logic for K-NN Recommendation ---
def get_course_recommendations(student_id):
    K = 6 
    try:
        student_index = utility_matrix.index.get_loc(student_id)
    except KeyError:
        return {"error": f"Student ID '{student_id}' not found."}, 404
    student_vector = utility_matrix.iloc[student_index, :].values.reshape(1, -1)
    _, indices = model_knn.kneighbors(student_vector, n_neighbors=K)
    neighbor_indices = indices.flatten()[1:]
    student_taken_courses = set(utility_matrix.iloc[student_index][utility_matrix.iloc[student_index] > 0].index)
    recommendation_counts = {}
    for i in neighbor_indices:
        neighbor_taken_courses = utility_matrix.iloc[i][utility_matrix.iloc[i] > 0].index
        for course in neighbor_taken_courses:
            if course not in student_taken_courses:
                recommendation_counts[course] = recommendation_counts.get(course, 0) + 1
    if not recommendation_counts:
        return {"message": "No new courses to recommend."}, 200
    sorted_recommendations = sorted(recommendation_counts.items(), key=lambda item: item[1], reverse=True)
    top_5 = [{'course_id': course, 'neighbor_count': count} for course, count in sorted_recommendations[:5]]
    return {"student_id": student_id, "recommendations": top_5}, 200

# --- Logic for XGBoost Performance Prediction ---
def predict_student_performance(data):
    try:
        df = pd.DataFrame([data])
        processed_data = preprocessor_xgb.transform(df)
        prediction_encoded = model_xgb.predict(processed_data)
        prediction_proba = model_xgb.predict_proba(processed_data)
        predicted_label = model_xgb.le_.inverse_transform(prediction_encoded)[0]
        probabilities = {label: float(prob) for label, prob in zip(model_xgb.le_.classes_, prediction_proba[0])}
        return {"predicted_outcome": predicted_label, "probabilities": probabilities}, 200
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}, 400

# --- Logic for Logistic Regression Dropout Prediction ---
def predict_dropout_risk(data):
    try:
        df = pd.DataFrame([data])
        processed_data = preprocessor_lr.transform(df)
        prediction_proba = model_lr.predict_proba(processed_data)
        dropout_risk = prediction_proba[0][1] 
        return {
            "student_id": data.get("student_id", "N/A"),
            "dropout_risk_probability": f"{dropout_risk * 100:.2f}%",
            "is_at_risk": bool(dropout_risk > 0.5)
        }, 200
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}, 400

# --- API Endpoints ---
@app.route('/recommend/<string:student_id>', methods=['GET'])
def recommend_courses_endpoint(student_id):
    response, status_code = get_course_recommendations(student_id)
    return jsonify(response), status_code

@app.route('/predict_performance', methods=['POST'])
def predict_performance_endpoint():
    response, status_code = predict_student_performance(request.get_json())
    return jsonify(response), status_code

@app.route('/predict_dropout', methods=['POST'])
def predict_dropout_endpoint():
    response, status_code = predict_dropout_risk(request.get_json())
    return jsonify(response), status_code

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
