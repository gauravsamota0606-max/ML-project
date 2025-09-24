from recommender.recommender import TravelRecommender

if __name__ == "__main__":
    recommender = TravelRecommender()
    recommender.load_data("data/destinations.csv", "data/reviews.csv")

    # Example user profile
    recommender.create_user_profile(user_id=1, preferences="Beach Adventure July")
    result = recommender.hybrid_recommend(user_id=1, preferences="Beach Adventure July")

    print("\nContent-Based Recommendations:")
    print(result["content_based"])

    print("\nCollaborative Recommendations:")
    print(result["collaborative"])
