"""Text cleaning shared by both datasets.

Mostly the same steps as the original notebook (clean_tweets), so that the
Twitter data and our Reddit data go through identical preprocessing.
Two things are different:
  * URLs anywhere in the text are removed (the original only checked the start
    of the tweet and dropped the whole tweet in that case);
  * no stemming. The original called PorterStemmer().stem() on the whole
    sentence, which only changed the last word, so we left it out.
"""
import re
import ftfy
from nltk.corpus import stopwords

# contraction list copied from the original notebook (shortened to the common ones)
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he's": "he is", "i'd": "i would",
    "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it's": "it is",
    "let's": "let us", "she's": "she is", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what's": "what is",
    "won't": "will not", "wouldn't": "would not", "you're": "you are", "you've": "you have",
}
_c_re = re.compile("(%s)" % "|".join(map(re.escape, CONTRACTIONS)))
_STOP = set(stopwords.words("english"))

URL_RE = re.compile(r"(\w+://\S+)|(www\.\S+)|(pic\.twitter\.com/\S+)|\bhttp\S*")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")        # usernames -> removed (privacy)
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
EMOJI_RE = re.compile(r"<Emoji:.*?>")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")


def clean_text(text):
    text = ftfy.fix_text(str(text)).lower()
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = _c_re.sub(lambda m: CONTRACTIONS[m.group(0)], text)
    text = re.sub(r"&amp;|\bamp\b", " ", text)    # html leftovers seen in the Reddit data
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [w for w in text.split() if w not in _STOP]
    return " ".join(words)
