import os
model_directory = '/home/haridoss/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'

if os.path.exists(model_directory):
    contents = os.listdir(model_directory)
    print("Directory exists. Contents:")
    print(contents)
else:
    print("Directory does not exist")