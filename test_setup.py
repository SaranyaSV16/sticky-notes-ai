# test_setup.py
# This script just checks that every library we need is installed and importable.

print("Testing imports...")

import flask
print("✅ Flask version:", flask.__version__)

from faster_whisper import WhisperModel
print("✅ faster-whisper imported successfully")

import sklearn
print("✅ scikit-learn version:", sklearn.__version__)

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("We need to submit the report by Friday.")
print("✅ spaCy loaded. Sample tokens:", [token.text for token in doc])

import nltk
from nltk.tokenize import sent_tokenize
print("✅ NLTK sentence split test:", sent_tokenize("Hello there. This is a test."))

import pandas as pd
print("✅ pandas version:", pd.__version__)

print("\n🎉 ALL LIBRARIES WORKING. Environment setup is complete.")