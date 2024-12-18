import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, f1_score

# Load the dataset into a DataFrame
df = pd.read_csv('US_Airlines_Twitter_Sentiment.csv')

# Visualize the number of tweets for each airline
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='airline')
plt.title('Number of Tweets for Each Airline')
plt.xlabel('Airline')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
#plt.show()

# Visualize the distribution of sentiments in the dataset
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='airline_sentiment')
plt.title('Distribution of Sentiments in the Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
#plt.show()

# Visualize the distribution of sentiments for each airline
plt.figure(figsize=(14, 8))
sns.countplot(data=df, x='airline', hue='airline_sentiment')
plt.title('Distribution of Sentiments for Each Airline')
plt.xlabel('Airline')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
#plt.show()

# Function to clean tweet text
def clean_tweet_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Apply the cleaning function to the tweet text
df['cleaned_text'] = df['text'].apply(clean_tweet_text)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Fit and transform the cleaned text
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])

# Convert the tfidf_matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())


# Define the features and target variable
X = tfidf_df
y = df['airline_sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Initialize the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute precision and F1 score
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("F1 Score:", f1)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()