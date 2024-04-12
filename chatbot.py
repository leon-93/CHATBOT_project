import streamlit as st
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Charger le modèle entraîné
model = load_model('chatbot_support_technique.h5')

# Charger le dataset
dataset = pd.read_csv('amazon_alexa.tsv', sep='\t')

# Dictionnaire de correspondance entre les classes et les messages
class_messages = dict(zip(dataset['feedback'], dataset['verified_reviews']))

# Définir et entraîner le tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['feedback'])

# Fonction pour prétraiter le texte
def preprocess_text(text):
    # Ajoutez ici votre logique de prétraitement
    # Par exemple : Convertir en minuscules, supprimer la ponctuation, tokenization, etc.
    return text

# Fonction pour prédire la classe d'une nouvelle question
MAX_SEQUENCE_LENGTH = 100
def predict_class(question):
    processed_question = preprocess_text(question)
    sequence = tokenizer.texts_to_sequences([processed_question])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = model.predict(padded_sequence)
    return prediction

# Interface utilisateur Streamlit
st.title('Technical Support Chatbot')

# Zone de texte pour saisir la question de l'utilisateur
user_question = st.text_input('Please enter your technical question:')

# Bouton pour soumettre la question
if st.button('Submit'):
    # Faire la prédiction
    predicted_class = predict_class(user_question)
    st.write('Predicted Class:', predicted_class)

    # Obtenir le message associé à la classe prédite
    predicted_message = class_messages.get(predicted_class)
    st.write('Predicted Message:', predicted_message)
