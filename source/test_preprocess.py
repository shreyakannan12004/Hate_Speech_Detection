from preprocess import clean_text

# Sample texts to test
samples = [
    "Go back to your country! @user http://spam.com",
    "I love my country",
    "You are stupid",
    "Have a nice day"
]

for text in samples:
    cleaned = clean_text(text)
    print("Original:", text)
    print("Cleaned: ", cleaned)
    print("---")
