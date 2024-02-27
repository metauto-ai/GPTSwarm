import json

input_file = '2023_validation_metadata.jsonl'
output_files = {
    1: 'level_1_val.jsonl',
    2: 'level_2_val.jsonl',
    3: 'level_3_val.jsonl'
}

data_by_level = {1: [], 2: [], 3: []}

with open(input_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        level = data.get("Level")
        if level in data_by_level:
            data_by_level[level].append(data)

for level, data_list in data_by_level.items():
    with open(output_files[level], 'w') as file:
        for data in data_list:
            json.dump(data, file)
            file.write('\n')
