import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load and preprocess the dataset
dataset = pd.read_csv("/product.csv")  # Replace with your dataset
dataset["Description"] = dataset["Description"].apply(lambda x: x.lower())  # Lowercase the descriptions

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the dataset descriptions
features = vectorizer.fit_transform(dataset["Description"])

def get_similar_items(payload):
    # Parse the JSON payload
    input_text = payload["text"]
    top_n = payload.get("top_n", 5)

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

    # Create the ranked suggestions
    suggestions = [{"rank": i + 1, "url": url} for i, url in enumerate(top_urls)]

    # Create the JSON response
    response = {
        "suggestions": suggestions
    }

    # Return the JSON response
    return json.dumps(response)

# Test the function with a sample payload
sample_payload = {
    "text": "Blue cotton t-shirt with logo",
    "top_n": 3
}
result = get_similar_items(sample_payload)
print(result)
