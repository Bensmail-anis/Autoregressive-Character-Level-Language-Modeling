def load_names(file_path):
    """Load names from a text file."""
    try:
        with open(file_path, 'r') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        return []