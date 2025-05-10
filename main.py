import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# download stopwords if not already
nltk.download('stopwords')

# Load dataset
df = pd.read_csv(r"C:\Users\verma\OneDrive\Desktop\New folder\train.csv")

# Dataset info
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(len(df))

# Label analysis
x = df.iloc[:, 2:].sum()
rowsums = df.iloc[:, 2:].sum(axis=1)
no_label_count = (rowsums == 0).sum()

print('Total number of comments:', len(df))
print('Total number of comments without labels:', no_label_count)
print('Total labels:', x.sum())

# Plot label counts
plt.figure(figsize=(6, 4))
sns.barplot(x=x.index, y=x.values, palette="muted")
plt.title('Label Counts')
plt.ylabel('Count')
plt.xlabel('Label')
plt.show()

# Plot labels per comment
plt.figure(figsize=(6, 4))
sns.countplot(x=rowsums.values, palette="muted")
plt.title('Labels per Comment')
plt.ylabel('# of Occurrences')
plt.xlabel('# of Labels')
plt.show()

# Drop 'id' column
df = df.drop(columns=['id'], axis=1)

# Text cleaning functions
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

def stemming(sentence):
    return " ".join([stemmer.stem(word) for word in sentence.split()])

# Apply cleaning
df['cleaned_text'] = df['comment_text'].apply(lambda x: stemming(remove_stopwords(clean_text(x))))

# Split data
X = df['cleaned_text']
y = df[df.columns[1:7]]  # Adjust if your labels are at different indices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Logistic Regression Pipeline
LR_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
])

# Define Naive Bayes Pipeline
NB_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', OneVsRestClassifier(MultinomialNB()))
])

# Function to run pipeline
def run_pipeline(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    for i, col in enumerate(y_test.columns):
        print(f"\nLabel: {col}")
        print(classification_report(y_test[col], y_pred[:, i]))

# Run both pipelines
print("Logistic Regression Results:")
run_pipeline(LR_pipeline, X_train, X_test, y_train, y_test)

print("\nNaive Bayes Results:")
run_pipeline(NB_pipeline, X_train, X_test, y_train, y_test)

labels = y_train.columns.values
labels

X_test.sample(1).values[0]
print(X_test.sample(1).values[0])

sentence = 'hello dick wikipedia fuckwhit ban'
stemmed_sentence = stemming(sentence)
results = LR_pipeline.predict([stemmed_sentence])[0]
for label, result in zip(labels, results):
    print("%14s %5s" % (label, result))

    sentence = 'hello how are you doing'
stemmed_sentence = stemming(sentence)
results = LR_pipeline.predict([stemmed_sentence])[0]
for label, result in zip(labels, results):
    print("%14s %5s" % (label, result))


def plot_roc_curve(test_labels, predict_prob):
    fpr, tpr, thresholds = roc_curve(test_labels, predict_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.legend(labels)
    
def plot_pipeline_roc_curve(pipeline, X_train, X_test, y_train, y_test):
    for label in labels:
        pipeline.fit(X_train, y_train[label])
        pred_probs = pipeline.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test[label], pred_probs)

plot_pipeline_roc_curve(LR_pipeline, X_train, X_test, y_train, y_test)
plt.show()
