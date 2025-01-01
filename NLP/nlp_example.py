import nltk
# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag

def basic_nlp_example(text):
    # 1. Sentence Tokenization
    sentences = sent_tokenize(text)
    print("Sentences:", sentences)

    # 2. Word Tokenization
    words = word_tokenize(text)
    print("\nWords:", words)

    # 3. Remove Stop Words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    print("\nFiltered Words (without stop words):", filtered_words)

    # 4. Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    print("\nStemmed Words:", stemmed_words)

    # 5. Part of Speech (POS) Tagging
    pos_tags = pos_tag(words)
    print("\nPOS Tags:", pos_tags)

# Example usage
text = "Natural language processing is fascinating. It helps computers understand human language."
basic_nlp_example(text)


# Text summarization example using gensim
""" from gensim.summarization.summarizer import summarize

def summarize_text(text, ratio=0.3):
    # Summarize text to 30% of original length
    summary = summarize(text, ratio=ratio)
    return summary

# Example for text classification using scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_text_classifier(training_texts, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(training_texts)
    classifier = MultinomialNB()
    classifier.fit(X, labels)
    return vectorizer, classifier
 """