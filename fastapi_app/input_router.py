from fastapi import APIRouter, HTTPException
from utils.utils import load_model, create_prompt_template
import os
import json

router = APIRouter()

@router.post("/input")
async def get_input(description: str):
    """
    Endpoint to receive input description and return a formatted prompt.
    """
    if not description:
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    # Load model and create prompt
    
    model = load_model()
    prompt_template = create_prompt_template()
    formatted_prompt = prompt_template.format(description=description)

    # Generate response
    response = model.invoke(formatted_prompt)

    # Save the response as a JSON file in the data folder

    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, "response.json")

    with open(file_path, "w") as json_file:
        json.dump({"response": response.content}, json_file)
    
    return {"response": response.content}