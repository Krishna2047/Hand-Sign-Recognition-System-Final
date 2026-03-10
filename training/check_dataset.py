import os

base = "dataset"

for folder in os.listdir(base):
    path = os.path.join(base, folder)
    print(folder, ":", len(os.listdir(path)), "images")
