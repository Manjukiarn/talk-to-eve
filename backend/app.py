from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pandas as pd
import numpy as np  # Import numpy for array manipulation

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize models and tools
analyzer = SentimentIntensityAnalyzer()  # VADER for sentiment analysis
nlp = spacy.load("en_core_web_sm")  # spaCy for keyword extraction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # BERT tokenizer
model = BertModel.from_pretrained('bert-base-uncased')  # BERT model

# Load and process the Twitter conversation dataset
def load_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Adjust encoding if necessary
    return data

# Function to process the dataset
def process_data(data):
    processed_data = data[['conversation_id', 'message', 'sentiment']].copy()  # Copy relevant columns
    processed_data.dropna(inplace=True)  # Remove rows with missing values
    return processed_data

# Load the dataset and process it
dataset = load_data(r'F:\talk-to-eve\backend\test.csv')  # Adjust the file path
processed_data = process_data(dataset)

# Example past conversations (you can use the processed dataset for RAG)
past_conversations = processed_data['message'].tolist()

# Function to embed text using BERT
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt")  # Tokenize the input text
    outputs = model(**inputs)  # Pass the input through the BERT model
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().squeeze()  # Ensure it's 2D

# Function to extract keywords using TF-IDF
def extract_keywords(text):
    doc = nlp(text)  # Process the text using spaCy
    preprocessed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])  # Lemmatization
    vectorizer = TfidfVectorizer(max_features=5)  # TF-IDF with a limit of 5 keywords
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])  # Apply TF-IDF
    keywords = vectorizer.get_feature_names_out()  # Extract keywords
    return keywords

# Function to map sentiment to RGB color
def get_sentiment_color(sentiment):
    if sentiment == 'positive':
        return (0, 255, 0)  # Green for positive
    elif sentiment == 'negative':
        return (255, 0, 0)  # Red for negative
    else:
        return (255, 255, 0)  # Yellow for neutral

# Retrieve the most relevant past conversation using cosine similarity
def retrieve_relevant_conversation(query):
    query_embedding = embed_text(query).reshape(1, -1)  # Ensure it's 2D for cosine similarity
    conversation_embeddings = [embed_text(conv).reshape(1, -1) for conv in past_conversations]  # Also ensure 2D

    # Stack conversation embeddings into a 2D array
    conversation_embeddings = np.vstack(conversation_embeddings)  # Combine into a single array
    similarities = cosine_similarity(query_embedding, conversation_embeddings)  # Calculate cosine similarities
    best_match_idx = similarities.argmax()  # Find the best match
    return past_conversations[best_match_idx]  # Return the best matching conversation

# Route to handle the query and perform various tasks
@app.route('/query', methods=['POST'])
def handle_query():
    
    user_input = request.json.get("text")  # Get the user input from the POST request
    if user_input is None:
        return jsonify({'error': 'No input provided!'}), 400  # Return an error response if input is None
    
    print(f"User input: {user_input}")

    # Sentiment analysis using VADER
    score = analyzer.polarity_scores(user_input)
    if score['compound'] >= 0.05:
        sentiment = 'positive'
    elif score['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    print(f"Sentiment determined: {sentiment}")

    # Extract keywords using TF-IDF
    keywords = extract_keywords(user_input)
    print(f"Extracted keywords: {keywords}")

    # Get RGB color based on sentiment
    rgb_color = get_sentiment_color(sentiment)
    print(f"RGB color determined: {rgb_color}")

    # RAG: Retrieve relevant conversation from the processed dataset
    relevant_conversation = retrieve_relevant_conversation(user_input)
    print(f"Relevant conversation retrieved: {relevant_conversation}")

    # Return the result as a JSON response
    return jsonify({
        'response': user_input,
        'sentiment': sentiment,
        'keywords': keywords.tolist(),
        'rgb_color': rgb_color,
        'relevant_conversation': relevant_conversation,
    })

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
