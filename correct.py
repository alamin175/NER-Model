import streamlit as st
import spacy_streamlit as spt
import spacy
from spacy.training.example import Example
from spacy.tokens import DocBin
import json

# Function to load the spaCy model
def load_model(model_path='en_core_web_sm'):
    return spacy.load(model_path)

# Initialize the model
nlp = load_model()

# Initialize the training data storage
if 'training_data' not in st.session_state:
    st.session_state.training_data = []

# Load training data from file
def load_training_data():
    try:
        with open("training_data.json", "r") as f:
            st.session_state.training_data = json.load(f)
    except FileNotFoundError:
        st.session_state.training_data = []

# Save training data to file
def save_training_data():
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f)

# Load training data when the app starts
load_training_data()

def preprocess_data(texts_and_labels):
    preprocessed_data = []
    for text, entities in texts_and_labels:
        ents = []
        for entity in entities['entities']:
            entity_text, label = entity[0], entity[1]
            start = text.find(entity_text)
            if start != -1:
                end = start + len(entity_text)
                # Ensure no overlapping entities
                if all(not (start < existing_end and end > existing_start) for existing_start, existing_end, _ in ents):
                    ents.append((start, end, label))
        preprocessed_data.append((text, {"entities": ents}))
    return preprocessed_data

def main():
    st.title('Named Entity Recognition (NER) App')

    # Initialize session state variables
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ''
    if 'entities' not in st.session_state:
        st.session_state.entities = []

    # Sidebar menu
    st.sidebar.title('Menu')
    home_button = st.sidebar.button('Home', key='home_button')
    ner_button = st.sidebar.button('NER', key='ner_button')
    train_button = st.sidebar.button('Train', key='train_button')
    view_data_button = st.sidebar.button('View Training Data', key='view_data_button')
    label_details_button = st.sidebar.button('NER Label Details', key='label_details_button')

    if home_button:
        st.session_state.page = 'home'
    elif ner_button:
        st.session_state.page = 'ner'
    elif train_button:
        st.session_state.page = 'train'
    elif view_data_button:
        st.session_state.page = 'view_data'
    elif label_details_button:
        st.session_state.page = 'label_details'

    # Set default page
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        st.header('Word Tokenization')
        st.session_state.raw_text = st.text_area(
            'Text to Tokenize', 
            st.session_state.raw_text, 
            placeholder='Enter text here...'
        )
        if st.button('Tokenize', key='tokenize_button'):
            if st.session_state.raw_text:
                docs = nlp(st.session_state.raw_text)
                spt.visualize_tokens(docs)
            else:
                st.warning('Please enter some text.')

    if st.session_state.page == 'ner':
        st.header('Named Entity Recognition')
        st.session_state.raw_text = st.text_area(
            'Text to Analyze', 
            st.session_state.raw_text, 
            placeholder='Enter text here to analyze...'
        )
        if st.button('Analyze', key='analyze_button'):
            if st.session_state.raw_text:
                docs = nlp(st.session_state.raw_text)
                st.session_state.entities = [(ent.text, ent.label_) for ent in docs.ents]
                spt.visualize_ner(docs)
            else:
                st.warning('Please enter some text.')
        
        if st.session_state.entities:
            st.subheader('Edit Entities')
            for i, (text, label) in enumerate(st.session_state.entities):
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input(f'Text {i+1}', value=text, key=f'text_{i}')
                with col2:
                    st.selectbox(f'Label {i+1}', options=nlp.get_pipe("ner").labels, index=nlp.get_pipe("ner").labels.index(label), key=f'label_{i}')
            
            if st.button('Update Entities', key='update_entities_button'):
                st.session_state.entities = [
                    (
                        st.session_state[f'text_{i}'],
                        st.session_state[f'label_{i}']
                    )
                    for i in range(len(st.session_state.entities))
                ]
                st.success('Entities updated!')

        st.subheader('Add to Training Data')
        if st.button('Add to Training Data', key='add_training_data_button'):
            entities = [(ent[0], ent[1]) for ent in st.session_state.entities]
            st.session_state.training_data.append((st.session_state.raw_text, {'entities': entities}))
            save_training_data()
            st.success('Text added to training data!')

    if st.session_state.page == 'train':
        st.header('Train NER Model')
        if st.button('Train', key='train_model_button'):
            if st.session_state.training_data:
                preprocessed_data = preprocess_data(st.session_state.training_data)
                train_model(preprocessed_data)
                st.success('Model trained successfully!')
                st.info('Reload the app to use the newly trained model.')
            else:
                st.warning('No training data available.')

    if st.session_state.page == 'view_data':
        st.header('View Training Data')
        if st.session_state.training_data:
            for i, (text, annotations) in enumerate(st.session_state.training_data):
                st.write(f"**Sample {i+1}**")
                st.write(f"Text: {text}")
                st.write(f"Entities: {annotations['entities']}")
        else:
            st.write("No training data available.")

    if st.session_state.page == 'label_details':
        st.header('NER Label Details')
        labels = {
            "MONEY": ("Monetary values, including units.", ["$5", "EUR 20"]),
            "NORP": ("Nationalities or religious or political groups.", ["American", "Christian", "Democrat"]),
            "ORDINAL": ("Ordinal numbers.", ["first", "second", "third"]),
            "ORG": ("Organizations.", ["Google", "United Nations"]),
            "PERCENT": ("Percentage values, including the '%' sign.", ["20%", "fifty percent"]),
            "PERSON": ("People, including fictional characters.", ["John Doe", "Sherlock Holmes"]),
            "PRODUCT": ("Objects, vehicles, foods, etc. (Not services).", ["iPhone", "Boeing 747"]),
            "QUANTITY": ("Measurements, as of weight or distance.", ["10 kilograms", "200 miles"]),
            "TIME": ("Times smaller than a day.", ["2 PM", "noon"]),
            "WORK_OF_ART": ("Titles of books, songs, etc.", ["Mona Lisa", "The Great Gatsby"])
        }

        for label, (description, examples) in labels.items():
            st.markdown(f'''
                <h3 style="color: #1E90FF;">{label}</h3>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Examples:</strong></p>
                <ul>
            ''', unsafe_allow_html=True)

            for example in examples:
                st.markdown(f'<li>{example}</li>', unsafe_allow_html=True)

            st.markdown('</ul><hr>', unsafe_allow_html=True)

    
def train_model(training_data):
    # Load a blank English model
    nlp = spacy.blank("en")
    
    # Create a new NER component
    ner = nlp.add_pipe("ner", last=True)

    # Add labels to the NER component
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Initialize the model with the training data
    optimizer = nlp.begin_training()

    # Convert training data to spacy format
    doc_bin = DocBin()
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        doc_bin.add(example.reference)

    # Training loop
    for i in range(10):  # Number of iterations
        losses = {}
        for text, annotations in training_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, losses=losses)
        st.write(f"Losses at iteration {i}: {losses}")

    # Save the model to disk
    nlp.to_disk("trained_model")


if __name__ == '__main__':
    # Add custom CSS to set a fixed width for the sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            width: 300px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    main()
