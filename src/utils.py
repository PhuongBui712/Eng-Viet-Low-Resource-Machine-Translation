

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip() for line in file.readlines()]

    return data