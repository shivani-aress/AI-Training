
from transformers import AutoTokenizer

# Step 1: Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Step 2: Define a hypothetical context window
context_limit = 100

# Step 3: Create sample text
text = """
Large Language Models are trained on large amounts of text.
They learn patterns in language and generate responses based
on the information provided to them.
"""

# Step 4: Repeat the text to increase its length
long_text = text * 20

# Step 5: Convert text into token IDs
tokens = tokenizer.encode(long_text)

# Step 6: Count the tokens
total_tokens = len(tokens)

# Step 7: Display the results
print("Context Window Limit:", context_limit)
print("Total Tokens:", total_tokens)

# Step 8: Check whether the text fits
if total_tokens <= context_limit:
    print("Status: Text fits within the context window")
else:
    print("Status: Context window limit exceeded")
