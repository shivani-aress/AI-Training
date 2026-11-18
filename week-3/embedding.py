from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love dogs",
    "Dogs are my favourite animals",
    "I enjoy eating pizza"
]

embeddings = model.encode(sentences)

print(embeddings)

print("Shape:", embeddings.shape)

print("Number of sentences:", embeddings.shape[0])

print("Dimensions per sentence:", embeddings.shape[1])

print("First sentence embedding:")
print(embeddings[0])

# Step 4: Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings)

# Step 5: Display the similarity matrix
print("Cosine Similarity:")
print(similarity_matrix)