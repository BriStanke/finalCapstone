# finalCapstone

## 1. A description of the dataset used.
I used an analysis dataset that contains product reviews from Amazon. This is a list
of over 34,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick,
and more. The dataset includes basic product information, rating, review text, and
more for each product.
## 2. Details of the preprocessing steps.
- The text data in the 'reviews.text' column is converted to lowercase and
whitespace is stripped to ensure consistency. Used .str.lower() and .str.strip().
- Dropping all missing values using .dropna.
- Stopwords and punctuation are removed from the text using .is_stop spaCy's
language processing capabilities.
- The resulting preprocessed text is stored in a new column named
'cleaned_reviews'.
## 3. Evaluation of results.
- Sentiment analysis is performed on the preprocessed text using the
spaCyTextBlob extension, which provides polarity scores for each review.
- The polarity score indicates the sentiment of each review, with positive values
representing positive sentiment, negative values representing negative
sentiment, and zero representing neutral sentiment.
- The similarity between pairs of reviews is computed using spaCy's similarity
function, which compares the semantic similarity between the preprocessed
documents.
## 4. Insights into the model's strengths and limitations.
### Strengths:
- Utilizes the spaCyTextBlob extension, which combines the power of spaCy for
natural language processing with TextBlob for sentiment analysis.
- Provides a quick and easy way to preprocess text data and perform sentiment
analysis without the need for extensive feature engineering.
- Incorporates error handling to handle cases where sentiment analysis cannot
be performed, ensuring robustness.
### Limitations:
- Preprocessing steps such as stopword removal and lemmatization may
oversimplify the text data, potentially losing important context or nuance.
- The accuracy of the sentiment analysis and similarity calculation heavily
depends on the quality of the preprocessed text and the capabilities of the
underlying spaCy models.
- The model may struggle with sarcasm, irony, or context-specific language that
can affect sentiment analysis accuracy.
