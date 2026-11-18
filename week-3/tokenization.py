
from transformers import AutoTokenizer

# Step 1: Load the pretrained GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Step 2: Define our input sentence
text = "I am learning Large Language Models"

# Step 3: Convert the sentence into tokens
tokens = tokenizer.tokenize(text)

# Step 4: Display the original sentence
print("Original Text:", text)

# Step 5: Display the tokens
print("Tokens:", tokens)

# Step 6: Display the total number of tokens
print("Number of Tokens:", len(tokens))

# Convert text directly into token IDs
token_ids = tokenizer.encode(text)

# Display the token IDs
print("Token IDs:", token_ids)

# Test an unfamiliar or technical word

new_text = "Neurocomputational"

# Tokenize the word
new_tokens = tokenizer.tokenize(new_text)

# Convert tokens into IDs
new_token_ids = tokenizer.encode(new_text)

print("Original Word:", new_text)
print("Tokens:", new_tokens)
print("Token IDs:", new_token_ids)
print("Number of Tokens:", len(new_tokens))