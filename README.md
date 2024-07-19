# Content-Based-Product-Recommendation-Chatbot-Project\
The chatbot uses product attributes to suggest relevant products to users.The chatbot involves data manipulation using pandas, finds similarity for recommendation, uses NLP pipeline using spacy for lemmatization,removing stop words, and removing punctuation, uses flask session management for using previous data to recommend content-based data.
Firstly installed the required dependencies like pandas, python, vs code, flask app, logging, string, spacy, sentence transformers, NLP pipeline, etc…
Then import these libraries and modules in your code editor. Used logging for debug the code.
Create a Flask web application instance using app = Flask(__name__) format.
Load the nlp pipeline model using spacy which is used to remove punctuation, remove stop words, lemmatization, remove the whitespaces.
Extract the attributes like (color, material, size, price, description) using nlp pipeline
Make preprocess text function for lemmatization , remove stop words, remove punctuation.
Load the Excel file of product dataset in csv form using pandas
Then remove whitespacing from starting and last of the string
Initialize the product texts and names
Convert to list
Function for extract no. of recommendations
Endpoint to process customer input and provide recommendations
Use post method and recommendations as endpoint for user interaction using api development tool i.e. (postman) to handle recommendations/ request.
Used try except method which includes:
Attempts to parse the JSON data sent in the POST request.
Checks if the JSON contains the key 'customer_input'. If not, raises a ValueError.
Extracts the value associated with the key 'customer_input' from the JSON data.
Preprocesses the customer input using a custom function preprocess_text.
Uses spaCy's NLP model to process the preprocessed text.
Initializes an empty list recommendations.
Loops through the product names. If a product name is found in the processed customer input, it adds the corresponding product description to the recommendations list.
If there are recommendations, returns the top recommendations as a JSON response.
If no recommendations are found, returns a message indicating that no suitable recommendations were found.
Catches and logs ValueError exceptions, returns a 400 Bad Request response with the error message.
Catches and logs any other exceptions, returns a 500 Internal Server Error response with the error message.
I used cosine similarity to calculate the similarity for recommendation of the product.
Results
Analysis: The chatbot effectively recommended products with high similarity scores, demonstrating the viability of content-based recommendations.
Discussion
Advantages: The system provides personalized recommendations without relying on user interaction data.
The need for searching becomes easier as the chat window is open for all types of queries.
Limitations: You can’t buy from chatbot. You only take info from chatbot for choosing a good type of product.
Future Work
Future improvements include refine recommendations according to user feedback.
Make the hybrid based product recommendation chatbot using large dataset
Make proper frontend.
