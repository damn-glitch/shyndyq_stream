import pickle

def fix_string_quote(data):
    fixed_data = [(item if isinstance(item, int) else item[2:-1]) for item in data]
    return fixed_data

def load_custom_pickle(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            if line.startswith('p'):
                data.append(int(line[1:]))
            elif line.startswith('aS'):
                data.append(line[1:])
    return data

def save_pickle(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Replace 'your_existing.txt' with the path to your existing custom format file
existing_file_path = 'C:\\Users\\tairk\\PycharmProjects\\Shyndyq\\complete\\Code\\dale-chall.pkl'

loaded_data = load_custom_pickle(existing_file_path)

if loaded_data:
    fixed_data = fix_string_quote(loaded_data)

    # Replace 'fixed_data.pkl' with the desired name for the new pickled file
    new_file_path = 'C:\\Users\\tairk\\PycharmProjects\\Shyndyq\\complete\\Code\\fixed_data.pkl'

    save_pickle(new_file_path, fixed_data)
    print(f"New pickled file '{new_file_path}' created with fixed data.")
else:
    print("Unable to fix pickled data. Please check the original file for issues.")