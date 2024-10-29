from llama_cpp import Llama
import llama_index
from text_splitter import split_text

# Load the Llama model from Hugging Face
llama = Llama.from_pretrained(
    repo_id="QuantFactory/Llama-3.2-1B-GGUF", filename="Llama-3.2-1B.Q2_K.gguf"
)

# Load the content from the Markdown file
markdown_path = "data/moby-dick-output.md"
with open(markdown_path, "r", encoding="utf-8") as file:
    content = file.read()

# Split the content into manageable chunks
chunks = split_text(content)

# Print the first 3 chunks to verify the output
print("First 3 chunks:")
for i, chunk in enumerate(chunks[10:13]):
    print(f"Chunk {i+1}:\n{chunk}\n")
