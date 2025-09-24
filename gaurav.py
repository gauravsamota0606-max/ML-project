import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class TravelRecommender:
    def __init__(self):
        self.destinations_df = None
        self.user_profiles = {}
    
    def load_data(self, destinations_file, reviews_file):
        # Load datasets
        self.destinations_df = pd.read_csv(destinations_file)
        self.reviews_df = pd.read_csv(reviews_file)
        # Preprocessing step (handle missing values, encoding etc.)
        self.destinations_df.fillna("", inplace=True)

    def create_user_profile(self, user_id, preferences):
        # Store user preferences in dictionary
        self.user_profiles[user_id] = preferences
    
    def content_based_recommend(self, preferences, n=5):
        # TF-IDF on activities + climate features
        self.destinations_df["combined"] = (
            self.destinations_df["climate"].astype(str) + " " +
            self.destinations_df["activities"].astype(str)
        )
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.destinations_df["combined"])

        # Convert preferences into same vector space
        pref_vec = vectorizer.transform([preferences])
        sim_scores = cosine_similarity(pref_vec, tfidf_matrix).flatten()

        top_idx = sim_scores.argsort()[::-1][:n]
        return self.destinations_df.iloc[top_idx][["destination", "climate", "activities"]]
    
    def collaborative_filter_recommend(self, user_id, n=5):
        # Very simple collab filter mock-up using user reviews
        user_ratings = self.reviews_df.pivot_table(
            index="user_id", columns="destination", values="rating"
        ).fillna(0)

        sim_matrix = cosine_similarity(user_ratings)
        sim_df = pd.DataFrame(sim_matrix, index=user_ratings.index, columns=user_ratings.index)

        similar_users = sim_df[user_id].sort_values(ascending=False).index[1:n+1]
        recommendations = user_ratings.loc[similar_users].mean().sort_values(ascending=False)[:n]
        return recommendations.index.tolist()
    
    def hybrid_recommend(self, user_id, preferences, n=5):
        content_rec = self.content_based_recommend(preferences, n)
        collab_rec = self.collaborative_filter_recommend(user_id, n)

        # Merge results (basic hybrid)
        return {
            "content_based": content_rec,
            "collaborative": collab_rec
        }
