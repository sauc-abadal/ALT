import json

def load_state(file_path):
    """Loads the state from the specified file."""
    global state_dict
    with open(file_path, "r") as f:
        state_dict = json.load(f)

def save_state(state_dict, file_path):
    """Saves the state to the specified file."""
    with open(file_path, "w") as f:
        json.dump(state_dict, f)
