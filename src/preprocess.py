import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

ps = PorterStemmer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [ps.stem(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]
    return " ".join(tokens)
