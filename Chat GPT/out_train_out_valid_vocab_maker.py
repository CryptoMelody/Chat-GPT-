import os
import lzma
from tqdm import tqdm
from google.colab import drive
drive.mount("/content/drive")

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files
folder_path =  "/content/drive/MyDrive/openwebtext"
output_file = "output{}.txt"
vocab_size = "vocab.txt"
output_file_train = "output_train.txt"
output_file_valid = "output_valid.txt"
files  = xz_files_in_dir(folder_path)
total_files = len(files)

vocab  = set()

split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_valid = files[split_index:]

with open(output_file_train, "w", encoding = "utf-8") as outfile:
  for filename in tqdm(files_train, total = len(files_train)):
    file_path = os.path.join(folder_path, filename)
    with lzma.open(file_path, "rt", encoding = "utf-8") as infile:
      text = infile.read()
      outfile.write(text)
      characters = set(text)
      vocab.update(characters)



with open(output_file_valid, "w", encoding = "utf-8") as outfile:
  for filename in tqdm(files_valid, total = len(files_valid)):
    file_path = os.path.join(folder_path, filename)
    with lzma.open(file_path, "rt", encoding = "utf-8") as infile:
      text = infile.read()
      outfile.write(text)
      chracters = set(text)
      vocab.update(characters)



with open(vocab_size, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + "\n")
