import datasets
import pandas as pd

# EDA
def ingest_data():
  ds = datasets.load_dataset("google-research-datasets/go_emotions")
  return ds

ds = ingest_data()

# Number of examples in each split
splits = ["test","train","validation"]
for split in splits:
  print(f"{split} : " +str(ds[split].num_rows))

# Available keys in features split
ds['test'].features.keys()

# Exploring data entries
print(ds['test'].to_pandas().head())
print(ds['test'].to_pandas().tail())

# All available labels and corresponding integer
label_list = ds['test'].features['labels'].feature.names
label_dict = {x:y for x,y in enumerate(label_list)}

# 5 examples from every emotion , class distribution
class_dist = {}
for i in range(28):
  example_text = [x for x in ds['test'] if i in x['labels']]
  class_dist[i] = len(example_text)
  # print("Emotion : " + label_dict[i])
  for j in example_text[:5]:
    # print(j['text'])

# Distribution of classes
class_dist

# Max , avg len of all
lenArray = []
maxLen = 0
avgLen = 0
for x in ds["test"]:
  lengthOfText = len(x['text'])
  maxLen = max(maxLen,lengthOfText)
  avgLen += lengthOfText
  lenArray.append(lengthOfText)
print(f"Max length = {maxLen}")
print(f"Average length = {avgLen/ds['test'].num_rows}")
plt.hist(lenArray)
