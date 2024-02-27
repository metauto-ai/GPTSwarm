import json

input_file = 'level_1_val.jsonl'
output_file = 'level_1_val.json'

data = []

with open(input_file, 'r') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)
