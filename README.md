# Clothing_Similarity_Search

This project provides a machine learning model for recommending similar clothing items based on their descriptions. Given an input text describing a clothing item, the model returns a ranked list of URLs to similar items from a dataset of clothing items.

## Requirements

- Python 3.9
- pandas
- scikit-learn

## Installation

1. Clone the repository:

git clone "https://github.com/ShakthiVar/Clothing_Similarity_Search"

2. Change to the project directory:

cd Clothing_Similarity_Search

3.Install The required dependencies

pip install -r requirements.txt

## USAGE

1.Prepare the dataset

-Product.csv

2.Train the model and save it as a pickle file.

3.Send a JSON payload with the input text and optional top_n value:

{
  "text": "Blue cotton t-shirt with logo",
  "top_n": 3
}

API Returns the JSON Response with the list of top matched URLs.

## CONTRIBUTION

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request.



Project Demo:https://youtu.be/eh4LMCRq_Jg
