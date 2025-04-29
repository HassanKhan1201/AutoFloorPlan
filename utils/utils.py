import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Set API Key if not already set
os.environ.setdefault(
    "GROQ_API_KEY",
    "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"
)

def load_model(model_name="llama3-8b-8192", provider="groq"):
    """
    Initializes and loads a chat model.

    Args:
        model_name (str): The name of the model to load.
        provider (str): The provider of the model.

    Returns:
        object: An instance of the initialized chat model.
    """
    return ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    )

def create_prompt_template():
    """
    Creates a PromptTemplate for real estate data extraction.

    Returns:
        PromptTemplate: Configured prompt template.
    """
    template = """
    You are an expert real estate data extractor.

    Extract structured information from the house description below and fill it strictly into the provided JSON format. 

    - Fill the missing thing based on your knowledge count >0.
    - Missing data count should not be 0.
    - Only return the final filled JSON response — no explanations, no formatting, no comments.
    - Do not include anything except the JSON.
    - Modify field names if necessary based on the content, but preserve the structure.
    - Do NOT repeat the prompt or include any introductory text.

    Here is the JSON format to fill:

    {{
    "house": {{
        "style": "<house style (e.g., modern, luxurious)>",
        "materials": "<materials used (e.g., glass, wood)>",
        "layout": "<layout description (e.g., open layout, everything visible)>",
        "lighting": "<lighting description (e.g., natural sunlight, photorealistic)>"
    }},
    "rooms": {{
        "bedrooms": {{
        "count": <number of bedrooms>,
        "description": "<bedroom description>"
        }},
        "bathrooms": {{
        "count": <number of bathrooms>,
        "description": "<bathroom description>"
        }}
    }},
    "garden": {{
        "description": "<garden description>"
    }},
    "swimming_pool": {{
        "description": "<swimming pool description>"
    }},
    "overall_scene": {{
        "visibility": "<visibility of areas>",
        "atmosphere": "<overall atmosphere>"
    }}
    }}

    Description:
    {description}
    """
    return template

def main():
    """
    Main execution function.
    """
    # Example house description
    description = """
    House with four bedrooms, 2 washrooms, one garden, and one swimming pool.
    All the things will be shown in one picture.
    """

    # Load model and create prompt
    model = load_model()
    prompt_template = create_prompt_template()
    formatted_prompt = prompt_template.format(description=description)

    # Generate and print response
    response = model.invoke(formatted_prompt)
    print(response.content)

if __name__ == "__main__":
    main()
