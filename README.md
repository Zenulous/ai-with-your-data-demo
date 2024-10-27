# Make your AI smarter and accurate with your own data

This demo belongs to an explainer [YouTube video](https://youtu.be/Pji4-9DKBUE)

## Installation Steps

1. **Clone the repository**

2. **Create a virtual environment (optional but recommended)**
   This project assumes you use Python 3.10.
   
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set environment variables**
    Create a .env file with the environment variables as denoted in .env.example
    In this example we use Azure CosmosDB

5. **Run the script**
    ```bash
    python ask_gpt.py 'what can you do?' no
    ```


Azure OpenAI can also be used, see https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints.
