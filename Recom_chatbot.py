from flask import Flask, request, jsonify
import pandas as pd
import logging
import string
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from sentence_transformers import SentenceTransformer
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load spaCy model for lemmatization and custom component
nlp = spacy.load('en_core_web_sm')

# Register custom attribute extension
if not Doc.has_extension("attributes"):
    Doc.set_extension("attributes", default={})

# Custom component for attribute extraction
@Language.component("attribute_extractor")
def extract_attributes(doc):
    attributes = {
        'material': None,
        'size': None,
        'color': None,
        'prod': None,
        'gender': None,
        'wears': None,
        'price': None,
        'description': None
    }
    # Add your custom attribute extraction logic here

    # Set the custom attributes to the Doc object
    doc._.attributes = attributes
    return doc

# Add the custom attribute extraction to the pipeline
nlp.add_pipe("attribute_extractor", last=True)

# Function to preprocess text using spaCy for lemmatization
def preprocess_text(text):
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_stop and token.lemma_ not in string.punctuation]
    return " ".join(lemmatized)

# Load product data from CSV (this part should remain)
product_data_df = pd.read_csv('Product_Data.csv', encoding='latin1')
product_data_df.columns = product_data_df.columns.str.strip() 
product_data_df['combined'] = product_data_df.apply(lambda row: f"{preprocess_text(str(row['prod']))}, {str(row['Material'])}, {str(row['Size'])}, {str(row['Color'])}, {str(row['Gender'])}, {str(row['Wears'])}, {str(row['Price'])}, {str(row['Description'])}", axis=1)

# Initialize product texts and names
product_texts = product_data_df['combined'].tolist()
product_names = product_data_df['prod'].tolist()

# Function to extract number of recommendations from input (dummy implementation)
def extract_num_recommendations(customer_input):
    return 5  # Default number of recommendations

# Endpoint to process customer input and provide recommendations
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Parse JSON request
        data = request.get_json()

        # Ensure the 'customer_input' field is present in the JSON
        if 'customer_input' not in data:
            raise ValueError("Invalid JSON format: 'customer_input' field is missing.")

        customer_input = data['customer_input']

        # Preprocess customer input
        customer_input_processed = preprocess_text(customer_input)
        doc = nlp(customer_input_processed)

        # Example logic to find recommendations based on product name
        recommendations = []
        for idx, name in enumerate(product_names):
            if name.lower() in customer_input_processed.lower():
                recommendations.append(product_texts[idx])

        if recommendations:
            return jsonify(recommendations[:extract_num_recommendations(customer_input)]), 200
        else:
            return jsonify({"message": "Sorry, we couldn't find a suitable recommendation based on your input. Please try again with more specific information."}), 200

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Error processing request")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
