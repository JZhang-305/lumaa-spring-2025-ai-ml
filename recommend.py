import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import argparse


# Create and configure a tf-idf vectorizer
def create_tfidf_vectorizer():
    return TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))


# Create tf-idf vectors from movie plots
def create_movie_vectors(df, vectorizer):
    return vectorizer.fit_transform(df["plot"].fillna(""))


# Get movie recommendations based on user description
def get_recommendations(
    user_description, df, vectorizer, tfidf_matrix, n_recommendations=5
):
    user_vector = vectorizer.transform([user_description])

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

    top_indices = similarity_scores[0].argsort()[-n_recommendations:][::-1]

    recommendations = []
    for idx in top_indices:
        title = df.iloc[idx]["title"]
        genre = (
            df.iloc[idx]["genre"] if "genre" in df.columns else "Genre not available"
        )
        score = similarity_scores[0][idx]
        recommendations.append((title, score, genre))

    return recommendations


# Print formatted recommendations
def print_recommendations(recommendations):
    print("\nTop Recommendations:\n")
    for i, (title, score, genre) in enumerate(recommendations, 1):
        print(f"{i}. {title.title()}")
        print(f"   Similarity Score: {score:.4f}")
        print(f"   Genre: {genre}")
        print()


# Create an interactive bar chart using plotly of movie recommendations
def visualize_recommendations(recommendations):
    titles = [rec[0].title() for rec in recommendations]
    scores = [rec[1] for rec in recommendations]
    genres = [rec[2] for rec in recommendations]

    # Create hover text with score and genre
    hover_text = [
        f"<b>{title}</b><br><br>"
        f"Similarity Score: {score:.4f}<br><br>"
        f"Genre: {genre}"
        for title, score, genre in zip(titles, scores, genres)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=titles,
                orientation="h",
                marker_color="rgb(30, 100, 255)",
                text=[f"{score:.4f}" for score in scores],
                textposition="auto",
                hovertext=hover_text,
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title="Movie Recommendations by Similarity Score",
        xaxis_title="Similarity Score",
        yaxis_title="Movie Title",
        yaxis={"categoryorder": "total ascending"},
        height=400 + (len(recommendations) * 30),
        margin=dict(t=30, l=10, r=10, b=10),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )

    fig.show()


parser = argparse.ArgumentParser()

# Arguments for user description and number of recommendations requested
parser.add_argument("description", type=str)
parser.add_argument("--num", type=int, default=5)
args = parser.parse_args()

# Load data and create vectors
df = pd.read_csv("movie_plots_trimmed.csv")
vectorizer = create_tfidf_vectorizer()
tfidf_matrix = create_movie_vectors(df, vectorizer)

# Get and print recommendations
recommendations = get_recommendations(
    args.description, df, vectorizer, tfidf_matrix, args.num
)
print_recommendations(recommendations)
visualize_recommendations(recommendations)

# Salary Expectation per month: $1500