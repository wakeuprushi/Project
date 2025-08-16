"""
Clubify Recommendation Engine

Description:
This is a personalized recommendation system that uses machine learning with K-means clustering 
to segment users based on their preferences and behaviors. It provides tailored recommendations 
through a Flask backend API by grouping similar users and suggesting popular items within their cluster.

Features:
- User segmentation with K-means clustering
- Personalized recommendations based on user clusters
- RESTful API built with Flask to serve recommendations
- Simple and extensible Python codebase

Dependencies:
- Flask
- pandas
- scikit-learn

Installation:
1. Install required libraries:
   pip install flask pandas scikit-learn
2. Prepare a CSV file named 'user_data.csv' with user features, e.g.:
   user_id,feature_1,feature_2,...,feature_n
3. Run this script:
   python app.py
4. Access recommendations at http://localhost:5000/recommend/<user_id>

Example CSV (user_data.csv):
user_id,likes_pop_music,likes_science_fiction,average_spent
1,5,2,300
2,3,5,150
3,4,4,220
...

"""

from flask import Flask, jsonify, request
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load user data and prepare clustering
data = pd.read_csv('user_data.csv')  # User data must have 'user_id' as first column
features = data.iloc[:, 1:]           # All columns except user_id as features

# Train K-means model
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(features)

def get_top_items_for_cluster(cluster_users):
    # Placeholder recommendation logic:
    # For demonstration, simply return the list of user_ids in the cluster.
    # In a real app, this might return popular items or user-preferred content.
    return cluster_users['user_id'].tolist()

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    user = data.loc[data['user_id'] == user_id]
    if user.empty:
        return jsonify({'error': 'User not found'}), 404
    cluster_label = int(user['cluster'].values[0])
    cluster_users = data[data['cluster'] == cluster_label]
    recommendations = get_top_items_for_cluster(cluster_users)
    return jsonify({
        'user_id': user_id,
        'cluster': cluster_label,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
