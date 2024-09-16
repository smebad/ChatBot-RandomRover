#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the necessary libraries
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configuring SSL context for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context

# Specify custom path for NLTK data and download necessary resources
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')  # Punkt tokenizer for sentence splitting


# In[2]:


# Defining the chatbot's intents, which represent possible topics of conversation
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hey", "Is anyone there?", "Hello", "How are you?", "What's up?"],
        "responses": ["Hello!", "Hi there, how can I help?", "Hey! How's it going?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye"],
        "responses": ["Goodbye!", "Take care! See you soon.", "It was nice talking to you!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thanks", "Thank you", "That's helpful", "I appreciate it"],
        "responses": ["You're welcome!", "Happy to help!", "Anytime!", "Glad I could assist!"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "Can you help me?", "What can you do?", "I need assistance"],
        "responses": ["Of course! How can I assist you?", "Tell me what you need help with.", "I'm here to help!"]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like today?", "How's the weather?", "What's the forecast?"],
        "responses": ["I don't have access to live weather data, but it's always good to check a trusted weather site!"]
    },
    {
        "tag": "date",
        "patterns": ["What's today's date?", "Tell me the date", "What day is it?"],
        "responses": [f"Today is 09/16/2024"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you?", "What's your age?"],
        "responses": ["I'm just a virtual being, but I'd say I'm timeless!", "Age is just a number, don't you think?"]
    },
    {
        "tag": "name",
        "patterns": ["What's your name?", "Who are you?", "What do I call you?"],
        "responses": ["I'm Random Rover, your helpful chatbot!", "Call me Rover, your virtual assistant!"]
    },
    {
        "tag": "default",
        "patterns": [""],
        "responses": ["I'm sorry, I don't understand. Could you rephrase?", "Can you clarify your question?"]
    }
]


#  ### Step 1: Build the chatbot's Natural Language Understanding (NLU) system ###
# * We need a way to convert user inputs (text) into a format that a machine learning model can process.
# * For this, I will use TF-IDF (Term Frequency-Inverse Document Frequency), which converts text into numerical values.

# In[3]:


# Create the vectorizer to transform text into numerical vectors
vectorizer = TfidfVectorizer()

# We will use Logistic Regression as the classifier for predicting which intent (tag) corresponds to user input
classifier = LogisticRegression(random_state=0, max_iter=10000)


# ### Step 2: Preparing the data for training
# * The model needs data to learn from, so we extract patterns from the intents and label them with the corresponding tags.

# In[4]:


# Preprocessing the data and looping it through each intent
X = []
y = []
for intent in intents:
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(intent["tag"])


# ### Step 3: Train the model
# * Now we have to fit the vectorizer on the patterns and train the classifier with the transformed inputs

# In[5]:


# Training the model
X_train = vectorizer.fit_transform(X)
y_train = y
classifier.fit(X_train, y_train)


# ### Step 4: Define the response generation logic
# * Creating a function that takes user input, transforms it, and predicts the best intent (tag) based on trained data.

# In[6]:


def generate_response(user_input):
    user_input = user_input.lower()  
    X_user = vectorizer.transform([user_input])  
    y_pred = classifier.predict(X_user)  
    tag = y_pred[0]  
    
    
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])  # Return a random response from the matched intent
    
    return "I'm sorry, I don't understand. Could you rephrase?"

# Defining the chatbot function that interacts with the user
def chatbot(user_input):
    return generate_response(user_input)


# ### Step 5: Building the Streamlit UI
# * The following function creates the user interface for interacting with the chatbot.

# In[7]:


# Define the chatbot response function
def chatbot(user_input):
    return generate_response(user_input)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def main():
    st.title("Random Rover - Your Personal Assistant")
    st.subheader("Feel free to ask me anything!")

    # Text input for user queries
    user_input = st.text_input("You:")

    # If user submits a question
    if user_input:
        response = chatbot(user_input)

        # Add the user's input and bot's response to the chat history
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Random Rover: {response}")

    # Display the chat history
    for message in st.session_state.chat_history:
        st.write(message)

    # Stop the app if the bot's response indicates the end of conversation
    if any([response.lower() in ['goodbye', 'thanks', 'bye'] for response in st.session_state.chat_history]):
        st.write("Thanks for chatting! Have a great day!")
        st.stop()

st.sidebar.title("Connect with me:")
st.sidebar.markdown(f"- [LinkedIn](https://lnkd.in/gxNURFgs)")
st.sidebar.markdown(f"- [GitHub](https://github.com/smebad)")
st.sidebar.markdown(f"- Email: mohammadebad1@hotmail.com")

# Entry point for Streamlit app
if __name__ == '__main__':
    main()

