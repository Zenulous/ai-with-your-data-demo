import os
from openai import OpenAI
import sys
from azure.cosmos import CosmosClient
from termcolor import colored
from dotenv import load_dotenv
import json
# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "The OpenAI API key is not set. Please set the environment variable 'OPENAI_API_KEY'."
    )

EMBEDDING_MODEL = "text-embedding-ada-002"

client = OpenAI(api_key=openai_api_key)

# Azure Cosmos DB configuration
cosmos_url = os.getenv("COSMOS_URL")
cosmos_key = os.getenv("COSMOS_KEY")
cosmos_database_name = 'diary_db'
cosmos_container_name = 'diary_embeddings'

cosmos_client = CosmosClient(cosmos_url, cosmos_key)
cosmos_container = cosmos_client.get_database_client(cosmos_database_name).get_container_client(cosmos_container_name)

def ask_gpt(question, context=None, model="gpt-4"):
    if context:
        system_prompt = f"You are an AI assistant. The following are relevant diary entries: {context}\nPlease answer the user's question based on the provided diary entries."
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

def find_most_relevant_entry(query_embedded, top_n=2):
    query = """
    SELECT TOP @top_n c.id, c.content, VectorDistance(c.embeddings, @query_embedding) AS SimilarityScore
    FROM c
    """
    parameters = [
        {"name": "@top_n", "value": top_n},
        {"name": "@query_embedding", "value": query_embedded},
    ]

    result = list(cosmos_container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))

    print(f"Most relevant entries with top_n {top_n}:")

    print(json.dumps(result, indent=4))

    if not result:
        return None
    
    result_text = "\n".join(entry['content'] for entry in result)
    return result_text


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

        print(colored(f"\nQuery embeddings generated.", "green"))
        relevant_context = find_most_relevant_entry(query_embedding)

        if relevant_context:
            print(colored(f"\nMost relevant diary entries content:", "blue"))
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
