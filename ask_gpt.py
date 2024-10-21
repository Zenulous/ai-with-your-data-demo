import os
from openai import OpenAI
import numpy as np
import sys
from termcolor import colored

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "The OpenAI API key is not set. Please set the environment variable 'OPENAI_API_KEY'."
    )

EMBEDDING_MODEL = "text-embedding-ada-002"

client = OpenAI(api_key=openai_api_key)


def ask_gpt(question, context=None, model="gpt-4"):
    if context:
        system_prompt = f"You are an AI assistant. The following is a relevant diary entry: {context}\nPlease answer the user's question based on the provided diary entry."
    else:
        system_prompt = "You are an AI assistant. Please answer the user's question."

    print(colored("\nSystem message (prompt for the GPT model):", "blue"))
    print(colored(system_prompt, "cyan"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def generate_openai_embeddings(input_text):
    print(
        colored(f"\nGenerating embeddings for the input text: '{input_text}'", "green")
    )
    response = client.embeddings.create(input=input_text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def load_embeddings(directory_path):
    print(colored(f"\nLoading embeddings from directory: {directory_path}", "green"))
    embeddings_dict = {}
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        with open(filepath, "r") as file:
            embeddings = [float(line.strip()) for line in file]
            embeddings_dict[filename] = embeddings
    return embeddings_dict


def find_most_relevant_entry(query_embedded, embeddings_dict):
    max_similarity = -1
    most_relevant_entry = None
    print(
        colored(
            "\nComparing query embeddings with stored embeddings to find the most relevant diary entry...",
            "green",
        )
    )
    for filename, embeddings in embeddings_dict.items():
        similarity = np.dot(query_embedded, embeddings) / (
            np.linalg.norm(query_embedded) * np.linalg.norm(embeddings)
        )
        print(colored(f"\nComparing with {filename}: {similarity:.4f}", "yellow"))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_entry = filename
        print(
            colored(
                f"New most relevant entry found: {filename} with similarity {similarity:.4f}",
                "magenta",
            )
        )
    return most_relevant_entry


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            colored(
                "Usage: python ask_gpt.py '<question>' <use_embeddings:yes/no>",
                "red",
            )
        )
        sys.exit(1)

    question = sys.argv[1]
    use_embeddings = sys.argv[2].lower() == "yes"

    print(colored(f"\nQuestion received: '{question}'", "green"))

    if use_embeddings:
        query_embedding = generate_openai_embeddings(question)

        embeddings_directory = "embeddings"
        embeddings_dict = load_embeddings(embeddings_directory)

        most_relevant_entry = find_most_relevant_entry(query_embedding, embeddings_dict)

        if most_relevant_entry:
            diary_directory = "data/diary"
            with open(
                os.path.join(
                    diary_directory, most_relevant_entry.replace(".embeddings", "")
                ),
                "r",
            ) as file:
                relevant_context = file.read()

            print(colored(f"\nMost relevant diary entry content:", "blue"))
            print(colored(relevant_context, "cyan"))

            gpt_response = ask_gpt(question, relevant_context)
            print(colored("\nGPT response:", "blue"))
            print(colored(gpt_response, "green"))
        else:
            print(colored("No relevant diary entry found.", "red"))
    else:
        gpt_response = ask_gpt(question)
        print(colored("\nGPT response:", "blue"))
        print(colored(gpt_response, "green"))
