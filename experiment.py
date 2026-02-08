import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

corpus = ['king is a strong man','queen is a wise woman','boy is a young man',
'girl is a young woman','prince is a young','prince will be strong',
'princess is young','man is strong','woman is pretty','prince is a boy',
'prince will be king','princess is a girl','princess will be queen']

statements_list = [word_tokenize(cor) for cor in corpus]
print("\nstatements_list =", statements_list)

stop_words = set(stopwords.words('english'))
documents = [[word for word in document if word.lower() not in stop_words] for document in statements_list]
print('\ndocuments =', documents)

vocabulary = FreqDist([word for document in documents for word in document])
print('\nvocabulary =', vocabulary)

vector1 = vocabulary['king']
print("\nking frequency =", vector1)

vector2 = vocabulary['man']
print("man frequency =", vector2)

sum_vector = vector1 + vector2
diff_vector = vector1 - vector2

print("\nsum vector =", sum_vector)
print("difference vector =", diff_vector)

similarity = (vocabulary['king'] * vocabulary['queen']) / (vocabulary['king'] ** 0.5 * vocabulary['queen'] ** 0.5)
print("\nCosine Similarity between 'king' and 'queen':", similarity)

most_similar = sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)
print("\nMost frequent words:", most_similar)

analogy_vector = vocabulary['king'] - vocabulary['man'] + vocabulary['woman']

def distance(item):
    return abs(item[1] - analogy_vector)

most_similar_analogy = sorted(vocabulary.items(), key=distance)[1]
print("\nAnalogy Result (king - man + woman):", most_similar_analogy)
