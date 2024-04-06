import json

def save_to_json_robust(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving file: {e}")

def load_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data
