
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1: Load tokenizer and embedding model

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 2: Company policy document

document = """
Employee Leave Policy:

All full-time employees are entitled to 20 paid leaves per year.
Employees must apply for planned leave at least 5 days in advance.
Sick leave can be taken when an employee is unwell.

Work From Home Policy:

Employees can work from home for 2 days per week.
Prior approval from the reporting manager is required.

Salary Policy:

Employees receive their salary on the last working day of every month.
Salary deductions may apply for unpaid leave.

Holiday Policy:

The company provides 12 public holidays every year.
The holiday calendar is shared with employees at the beginning of the year.
"""


# STEP 3: Convert document into token IDs

tokens = tokenizer.encode(document, add_special_tokens=False)


# STEP 4: Create chunks

chunk_size = 50
overlap = 10

chunks = []

start = 0

while start < len(tokens):

    end = start + chunk_size

    chunk_tokens = tokens[start:end]

    chunk_text = tokenizer.decode(chunk_tokens)

    chunks.append(chunk_text)

    start += chunk_size - overlap


print("Total chunks created:", len(chunks))


# STEP 5: Generate embeddings for all chunks

chunk_embeddings = model.encode(chunks)

print("Chunk embeddings shape:", chunk_embeddings.shape)


# STEP 6: Ask a question

query = "How many paid leaves do employees get?"


# STEP 7: Generate embedding for the question

query_embedding = model.encode([query])


# STEP 8: Compare question embedding with chunk embeddings

similarity_scores = cosine_similarity(
    query_embedding,
    chunk_embeddings
)[0]


# STEP 9: Rank chunks by similarity

top_k = 2

top_indices = similarity_scores.argsort()[::-1][:top_k]


# STEP 10: Display retrieved chunks

print("\nUser Question:", query)

print("\nRetrieved Information:")

for index in top_indices:

    print("\nSimilarity Score:", similarity_scores[index])

    print("Retrieved Chunk:")
    print(chunks[index])
