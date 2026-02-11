import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Sports corpus
corpus = [
    'basketball is a sport played with a ball',
    'soccer is played by two teams on a field',
    'football involves a lot of physical contact',
    'athletes train hard to improve their performance',
    'coaching is an important part of every sport',
    'basketball players need good coordination',
    'a team consists of players and a coach',
    'training and exercise are important for health',
    'soccer players score goals to win games',
    'football teams compete in leagues and tournaments'
]

# Step 2: Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
words = vectorizer.get_feature_names_out()

print("Vocabulary Words:\n", words)

# Step 3: PCA Dimensionality Reduction
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(X.T)

# Step 4: Find Similar Words
def find_similar_words(target_word, word_vectors, words, top_n=5):
    if target_word not in words:
        return f"{target_word} not found"

    target_idx = np.where(words == target_word)[0][0]
    target_vector = word_vectors[target_idx].reshape(1, -1)

    similarities = cosine_similarity(target_vector, word_vectors)[0]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]

    return [words[i] for i in similar_indices]

input_word = "basketball"
similar_words = find_similar_words(input_word, word_vectors_2d, words)

print("\nSimilar words to", input_word, ":", similar_words)

# Step 5: Visualization
plt.figure(figsize=(8, 8))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

for i, word in enumerate(words):
    plt.text(word_vectors_2d[i, 0], word_vectors_2d[i, 1], word)

plt.title("Word Embeddings Visualization using PCA")
plt.show()
