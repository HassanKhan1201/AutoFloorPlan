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
    Creates a PromptTemplate for extracting spatially structured real estate data with coordinates
    for easy drawing using OpenCV.

    Returns:
        PromptTemplate: Configured prompt template.
    """
    template = """
    You are an expert real estate spatial data extractor.

    Extract structured information including entities and their spatial coordinates from the house description below.
    The extracted information will be used to draw the house layout using OpenCV, so provide coordinates and dimensions
    that can be directly translated to image coordinates (pixel-based).

    - Fill missing values based on context or common knowledge. Do NOT leave any fields empty.
    - Coordinates should be in the format [x, y].
    - Dimensions (width and height) should be provided in meters, which will later be scaled to pixels for drawing.
    - Return ONLY the final filled JSON. No explanations or comments.
    - Do NOT repeat the prompt or include any introduction.
    - If coordinates or dimensions are implied, estimate them reasonably.

    Here is the JSON format to fill:

    {{
    "house": {{
        "style": "<house style (e.g., modern, luxurious)>",
        "materials": "<materials used (e.g., glass, wood, concrete)>",
        "layout": "<layout description (e.g., open concept, linear)>",
        "lighting": "<lighting description (e.g., natural, ambient, warm)>"
    }},
    "entities": [
        {{
            "type": "bedroom",
            "name": "Bedroom 1",
            "coordinates": [<x>, <y>],  # top-left corner in pixels
            "dimensions_m": [<width>, <height>],  # in meters, later to be scaled to pixels
            "description": "<e.g., master bedroom with balcony>"
        }},
        {{
            "type": "bathroom",
            "name": "Bathroom 1",
            "coordinates": [<x>, <y>],
            "dimensions_m": [<width>, <height>],
            "description": "<e.g., modern bathroom with glass shower>"
        }},
        {{
            "type": "kitchen",
            "name": "Kitchen",
            "coordinates": [<x>, <y>],
            "dimensions_m": [<width>, <height>],
            "description": "<e.g., open kitchen with island counter>"
        }},
        {{
            "type": "living_room",
            "name": "Living Room",
            "coordinates": [<x>, <y>],
            "dimensions_m": [<width>, <height>],
            "description": "<e.g., spacious living room with TV unit>"
        }},
        {{
            "type": "garden",
            "name": "Garden",
            "coordinates": [<x>, <y>],
            "dimensions_m": [<width>, <height>],
            "description": "<e.g., lush green garden with stone pathway>"
        }},
        {{
            "type": "staircase",
            "name": "Staircase",
            "coordinates": [<x>, <y>],
            "dimensions_m": [<width>, <height>],
            "description": "<e.g., staircase leading to upper floor>"
        }}
        // Add more entities as required
    ],
    "overall_scene": {{
        "visibility": "<what areas are visible from where>",
        "atmosphere": "<e.g., peaceful, luxurious, cozy>"
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
