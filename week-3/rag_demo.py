
from transformers import AutoTokenizer

# Step 1: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Step 2: Sample company policy document
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

# Step 3: Convert document into token IDs
tokens = tokenizer.encode(document, add_special_tokens=False)

# Step 4: Define chunk settings
chunk_size = 50
overlap = 10

# Step 5: Create chunks
chunks = []

start = 0

while start < len(tokens):

    end = start + chunk_size

    chunk_tokens = tokens[start:end]

    chunk_text = tokenizer.decode(chunk_tokens)

    chunks.append(chunk_text)

    start += chunk_size - overlap

# Step 6: Display chunks
for i, chunk in enumerate(chunks):

    print(f"\nChunk {i + 1}:")
    print(chunk)

    print("Token count:", len(tokenizer.encode(
        chunk, add_special_tokens=False
    )))
