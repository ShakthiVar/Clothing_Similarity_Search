import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the dataset descriptions
features = vectorizer.fit_transform(dataset["Description"])


def get_similar_items(input_text, top_n=5):
    # Preprocess the input text
    input_text = input_text.lower()

    # Transform the input text into TF-IDF features
    input_features = vectorizer.transform([input_text])

    # Calculate the cosine similarity between the input text and dataset descriptions
    similarities = cosine_similarity(input_features, features).flatten()

    # Get the indices of top-N most similar items
    top_indices = similarities.argsort()[::-1][:top_n]

    # Get the URLs of the top-N most similar items
    top_urls = dataset.loc[top_indices, "Product_URL"].tolist()

    # Return the ranked suggestions as JSON responses
    suggestions = [{"rank": i + 1, "url": url} for i, url in enumerate(top_urls)]
    return {"suggestions": suggestions}

# Load the model from the pickle file
with open("C:/Users/shakt/Documents/app/clothing_model.pkl", "rb") as file:
    model = pickle.load(file)

# Test the loaded model
input_text = "Blue cotton t-shirt with logo"
result = model(input_text, top_n=5)
print(result)