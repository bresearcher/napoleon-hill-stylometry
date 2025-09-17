orensic Stylometry of the Napoleon Hill Corpus
This repository contains the data and code used for the computational analysis referenced in the book "THE DEVIL'S MESSENGER: Cataplexis, Cognitive Warfare, and the Inoculation of Truth."

The analysis was conducted to empirically test the "Paradigm Break" hypothesis concerning Napoleon Hill's 1938 manuscript, Outwitting the Devil. The script analyze.py processes the corpus of text files to generate the stylometric and sentiment data discussed in Appendix B of the book.

The purpose of this public repository is to ensure reproducibility, allowing any researcher to verify the findings by running the analysis on the original source data.

Data Corpus
The corpus consists of four plain text files, which have been cleaned to remove publisher-added content:

law_of_success.txt

think_and_grow_rich.txt

outwitting_the_devil.txt

later_voice_composite.txt

Requirements
This script was run using Python 3. To install the necessary libraries, run the following command in your terminal:

pip install nltk pandas scikit-learn matplotlib textblob

You will also need to download the 'punkt' tokenizer for the NLTK library. This can be done by running the following commands in a Python interpreter:

import nltk
nltk.download('punkt')

With all .txt files and the analyze.py script in the same directory, and after installing the required libraries, run the script from your terminal:


python3 analyze.py

The script will print the results tables to the console and save a Principal Components Analysis (PCA) plot as stylometric_analysis_pca.png in the same directory.
