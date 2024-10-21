import os
import openai

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "The OpenAI API key is not set. Please set the environment variable 'OPENAI_API_KEY'."
    )
openai.api_key = openai_api_key

EMBEDDING_MODEL = "text-embedding-ada-002"

client = openai.OpenAI()


def generate_openai_embeddings(input_text):
    response = client.embeddings.create(input=input_text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def process_diary_entries(directory_path):
    embeddings_dict = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                content = file.read()
                print(f"Generating embeddings for {filename}...")
                embeddings = generate_openai_embeddings(content)
                embeddings_dict[filename] = embeddings

    save_embeddings(embeddings_dict)


def save_embeddings(embeddings_dict):
    output_directory = "embeddings"
    os.makedirs(output_directory, exist_ok=True)
    for filename, embeddings in embeddings_dict.items():
        output_path = os.path.join(output_directory, filename + ".embeddings")
        with open(output_path, "w") as file:
            for value in embeddings:
                file.write(f"{value}\n")


if __name__ == "__main__":
    diary_directory = "data/diary"
    process_diary_entries(diary_directory)
    print("Embeddings generated and saved successfully.")
