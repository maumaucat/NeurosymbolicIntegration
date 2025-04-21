import numpy as np

def parse_iris_data(iris_file):
    with open(iris_file, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            if line.startswith('Id'):
                continue
            content =  line.strip().split(',')
            attributes = content[1:-1]
            label = content[-1]
            data.append((attributes, label))

    return data

def get_attributes(iris_file):
    data = parse_iris_data(iris_file)
    att_data = [[x[0]] for x in data]
    return np.array(att_data).reshape(len(att_data), 4).astype(np.float32)

def get_labels(iris_file):
    data = parse_iris_data(iris_file)
    label_data = [[x[1]] for x in data]
    return np.array(label_data).reshape(len(label_data), 1)


