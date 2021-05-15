import os

def get_filenames_in_directory(directory):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        f.extend(filenames)
        break
    return f