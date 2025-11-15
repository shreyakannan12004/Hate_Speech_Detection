import re
import string
from nltk.corpus import stopwords

# Load stopwords (only one time)
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [tok for tok in tokens if tok not in stop_words and len(tok) > 1]

    # Return cleaned text
    return " ".join(tokens)
