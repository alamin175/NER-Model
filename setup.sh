#!/bin/bash

# Update package list
sudo apt-get update

# Install Python packages
pip install spacy
pip install spacy_streamlit
pip install streamlit
pip install thinc==8.1.0

# Install spaCy model
python -m spacy download en-core-web-sm
