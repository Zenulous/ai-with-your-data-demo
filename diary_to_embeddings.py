import os
import openai
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "The OpenAI API key is not set. Please set the environment variable 'OPENAI_API_KEY'."
    )
openai.api_key = openai_api_key

EMBEDDING_MODEL = "text-embedding-ada-002"

client = openai.OpenAI()

cosmos_url = os.getenv("COSMOS_URL")
cosmos_key = os.getenv("COSMOS_KEY")
cosmos_database_name = "diary_db"
cosmos_container_name = "diary_embeddings"

cosmos_client = CosmosClient(cosmos_url, cosmos_key)

try:
    cosmos_database = cosmos_client.create_database_if_not_exists(
        id=cosmos_database_name
    )
except exceptions.CosmosResourceExistsError:
    cosmos_database = cosmos_client.get_database_client(cosmos_database_name)

indexing_policy = {
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?', "path": "/embeddings/*"}],
    "vectorIndexes": [{"path": "/embeddings", "type": "quantizedFlat"}],
}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embeddings",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 1536,  # This is the amount of dimensions for the text-embedding-ada-002 model
        }
    ]
}

try:
    cosmos_container = cosmos_database.create_container_if_not_exists(
        id=cosmos_container_name,
        partition_key=PartitionKey(path="/id"),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
    )
    print(f"Container {cosmos_container_name} created")
except exceptions.CosmosHttpResponseError as e:
    print(f"Error creating container: {str(e)}")


def generate_openai_embeddings(input_text):
    response = client.embeddings.create(input=input_text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def process_diary_entries(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                content = file.read()
                print(f"Generating embeddings for {filename}...")
                embeddings = generate_openai_embeddings(content)

                document = {
                    "id": filename,
                    "content": content,
                    "category": "dailyDiary",
                    "embeddings": embeddings,
                }
                cosmos_container.upsert_item(body=document)


if __name__ == "__main__":
    diary_directory = "data/diary"
    process_diary_entries(diary_directory)
    print("Embeddings generated and saved successfully to Cosmos DB.")
