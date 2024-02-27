import os

prefix = "2023_validation_"

files_and_dirs = os.listdir('.')

for name in files_and_dirs:
    if os.path.isfile(name) and name.startswith(prefix):
        new_name = name[len(prefix):]
        
        os.rename(name, new_name)
        print(f"Renamed '{name}' to '{new_name}'")
