def individual_serial(prompt) -> dict:
    return {
        "id": str(prompt['_id']),
        "name": str(prompt['name']),
        "description": str(prompt['description']),
        "content": str(prompt['content']),
    }


def list_serial(prompts) -> list:
    return [individual_serial(prompt) for prompt in prompts]
