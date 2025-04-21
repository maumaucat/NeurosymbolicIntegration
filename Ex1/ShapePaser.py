import ast
import json
from pathlib import Path

# Function to load data from the given file and returns a list of objects
def load_data(filepath) -> []:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        content = "".join(content[1:-3])
        # Splitte den Inhalt in einzelne Objekte, basierend auf dem Muster '}, {'
        objects = []
        for entry in content.split("},"):
            entry = entry + "}"
            obeject = ast.literal_eval(entry)
            objects.append(obeject)
        return objects

# Function to get the size of a given object
def get_size(object : dict) -> str:
    return object['Predicates'][1]

# Function to get the position of a given object
def get_position(object : dict) -> []:
    return object['Tags']

# Function to get the shape of a given object
def get_shape(object : dict) -> str:
    return object['Predicates'][0]

# Function to get the lable of a given object
def get_label(object : dict) -> str:
    return object['Consts'][0]

# Example usage / test
folder = Path("C:/Users/ms/Documents/Uni/Master/NeurosymbolischeIntegration/dev2/Datasets/shapes/test")
for file in folder.iterdir():
    if file.is_file():
        print("="*50)
        print(f"File: {file.name}")
        for object in load_data(file):
            print("-"*50)
            print("Shape: ", get_shape(object))
            print("Label: ", get_label(object))
            print("Size: ", get_size(object))
            print("Position: ", get_position(object))