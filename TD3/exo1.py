import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def cleanWord(tweets):
        nltk.download('punkt_tab')
        tweets = tweets.translate(str.maketrans('', '', string.punctuation))
        tweets = nltk.word_tokenize(tweets)
        tweets = [word.lower() for word in tweets]
        stopwords = nltk.corpus.stopwords.words('english')
        tweets = [word for word in tweets if word not in stopwords]
        stemmer = nltk.stem.PorterStemmer()
        tweets = [stemmer.stem(word) for word in tweets]
        return str(tweets)


if __name__ == '__main__':
    with open('tweets.txt', 'r', encoding='utf-8') as file:
            tweets = file.read()
            texte = cleanWord(tweets)
            print(texte)