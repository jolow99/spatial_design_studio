import os
from pathlib import Path

def simplify_filenames(directory):
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                path = Path(root) / file
                
                # Extract the base pattern (curved or rect) and number
                if 'curved' in file.lower():
                    # Find the number after 'curved'
                    number = ''.join(filter(str.isdigit, file.split('curved')[1].split('_')[0]))
                    new_name = f'curved{number}.csv'
                elif 'rect' in file.lower():
                    # Find the number after 'rect'
                    number = ''.join(filter(str.isdigit, file.split('rect')[1].split('_')[0]))
                    new_name = f'rect{number}.csv'
                else:
                    continue
                
                # Create the new path
                new_path = path.parent / new_name
                
                # Rename the file
                try:
                    path.rename(new_path)
                    print(f'Renamed: {file} -> {new_name}')
                except Exception as e:
                    print(f'Error renaming {file}: {e}')

# Usage example
directory = '../data/expert/subject_2'
simplify_filenames(directory)
