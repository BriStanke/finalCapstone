# Importing necessary libraries
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import random

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Load the dataset
amazon_data = pd.read_csv("amazon_product_reviews.csv")

# Investigate the dataset by displaying 5 first rows
amazon_data.head()
# Getting all the column names
amazon_data.columns
# Investigating the dataset to understand the number of rows, the number of columns, and identifying columns with missing values
amazon_data.info()

# Convert text to lowercase and strip whitespace
amazon_data['reviews.text'] = amazon_data['reviews.text'].str.lower().str.strip()

# Dropping all missing values from 'reviews.text' column
clean_data = amazon_data.dropna(subset=['reviews.text'])
clean_data.isnull().sum()

# Defining a function to preprocessing the text data
def preprocess_text(text):
    # Remove stopwords by using .is_stop attribute in spaCy
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Apply preprocessing to 'reviews.text' column (this takes longer with better accuracy)
# Change column name from 'reviews.text' to 'cleaned_reviews'
# clean_data['cleaned_reviews'] = clean_data['reviews.text'].apply(preprocess_text)

# Apply preprocessing to 'reviews.text' column (this will be quicker and with less accuracy)
# Only use a small subset (1000) of the data
# Change column name from 'reviews.text' to 'cleaned_reviews'
clean_data = clean_data.sample(1000, random_state=42)
clean_data['cleaned_reviews'] = clean_data['reviews.text'].apply(preprocess_text)

# Check the first few rows of the cleaned data column 'cleaned_reviews'
print(clean_data['cleaned_reviews'].head())
# Display number of rows we will be working
print(clean_data['cleaned_reviews'].shape)

# Select column that we will be working on
reviews_data = clean_data['cleaned_reviews']

# Function for sentiment analysis
def analyze_sentiment(review):
    doc = nlp(review)
    # Adding polarity attribute from TextBlob library
    blob = doc._.blob
    if blob is not None:
        polarity = blob.polarity
        # A polarity score of 1 indicates a very positive sentiment, while a polarity score of -1 indicates a very negative sentiment. A polarity score of 0 indicates a neutral sentiment.
        if polarity > 0:
            sentiment = 'positive'
        elif polarity < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return sentiment, polarity
    else:
        # Error handling case where sentiment analysis couldn't be performed
        return 'neutral', 0.0


# Test sentiment analysis function by taking random data with .iloc
sample_review = reviews_data.sample(n=1).iloc[0]
sentiment, polarity = analyze_sentiment(sample_review)
# Output random review
print(f"\nSample Review:", sample_review)
# Output sentiment
print(f"Sentiment:", sentiment)
# Output polarity score
print(f"Polarity Score: {polarity:.2f}")



# Choose two random product reviews
random_reviews = reviews_data.sample(n=2, random_state=42)

# Process the reviews using SpaCy
doc1 = nlp(random_reviews.iloc[0])
doc2 = nlp(random_reviews.iloc[1])

# Compute similarity between the two product reviews
similarity_score = doc1.similarity(doc2)
# Output 2 random reviews
print(f"\nFirst review :", random_reviews.iloc[0])
print(f"Second review :", random_reviews.iloc[1])

# Print the similarity score between 2 random reviews
# A similarity score of 1 indicates that the two reviews are more similar, while a similarity score of 0 indicates that the two reviews are not similar
print(f"Similarity between the two reviews: {similarity_score:.2f}")