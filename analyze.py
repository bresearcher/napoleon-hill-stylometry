import sys
print(f"--- This script is running from: {sys.executable} ---")
import os
import pandas as pd
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def calculate_metrics(text):
    """Calculates stylistic and sentiment metrics for a given text."""
    # Sentence tokenization
    sentences = nltk.sent_tokenize(text)
    # Word tokenization
    words = nltk.word_tokenize(text.lower())
    
    # 1. Sentence Complexity (Average Sentence Length)
    if len(sentences) > 0:
        avg_sentence_length = len(words) / len(sentences)
    else:
        avg_sentence_length = 0
        
    # 2. Lexical Diversity (Type-Token Ratio)
    if len(words) > 0:
        ttr = len(set(words)) / len(words)
    else:
        ttr = 0
        
    # 3. Sentiment Analysis
    sentiment = TextBlob(text)
    polarity = sentiment.sentiment.polarity
    subjectivity = sentiment.sentiment.subjectivity
    
    return {
        'Avg Sentence Length': avg_sentence_length,
        'Lexical Diversity (TTR)': ttr,
        'Sentiment Polarity': polarity,
        'Sentiment Subjectivity': subjectivity
    }

def main():
    """Main function to run the analysis."""
    # --- CONFIGURATION ---
    # IMPORTANT: List your four text file names here
    filenames = [
        'law_of_success.txt',
        'think_and_grow_rich.txt',
        'outwitting_the_devil.txt',
        'later_voice_composite.txt'
    ]
    # --- END CONFIGURATION ---

    # Check if files exist
    if not all(os.path.exists(f) for f in filenames):
        print("Error: Make sure all your text files are in the same folder as this script")
        print("and their names match the 'filenames' list in the code.")
        return

    # Process each file and gather data
    all_metrics = []
    corpus = []
    clean_filenames = []

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
            corpus.append(text)
            metrics = calculate_metrics(text)
            all_metrics.append(metrics)
            # Use a cleaner name for the plot labels
            clean_filenames.append(os.path.splitext(filename)[0].replace('_', ' ').title())

    # Create a DataFrame for the basic metrics
    metrics_df = pd.DataFrame(all_metrics, index=clean_filenames)
    
    # --- Advanced Stylistic Analysis (TF-IDF and PCA) ---
    # This creates a stylistic fingerprint based on word frequency
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(corpus).toarray()
    
    # Combine TF-IDF features with our other metrics
    combined_features = pd.concat([metrics_df.reset_index(drop=True), pd.DataFrame(X)], axis=1)
    combined_features.columns = combined_features.columns.astype(str)
    # Scale the features for PCA
    X_scaled = StandardScaler().fit_transform(combined_features)
    
    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'], index=clean_filenames)
    
    # --- Display and Visualize Results ---
    print("--- Stylistic and Sentiment Metrics ---")
    print(metrics_df)
    print("\n")

    print("--- PCA Results ---")
    print("This shows the coordinates of each book on the style plot.")
    print(pca_df)
    print("\n")

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_df['PC 1'], pca_df['PC 2'], s=100)
    
    for i, title in enumerate(pca_df.index):
        plt.annotate(title, (pca_df['PC 1'][i], pca_df['PC 2'][i]), fontsize=12, ha='right')
        
    plt.title('Stylistic Analysis of Napoleon Hill\'s Works (PCA)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True)
    
    # Save the plot to a file
    plot_filename = 'stylistic_analysis_pca.png'
    plt.savefig(plot_filename)
    
    print(f"Success! The analysis is complete.")
    print(f"The results table is printed above, and the visualization plot has been saved as '{plot_filename}'")

if __name__ == '__main__':
    main()
