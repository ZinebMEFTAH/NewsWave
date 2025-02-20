import csv
from google.colab import files

# Open the input file in read mode and output file in write mode
with open('test.csv', 'r') as inp, open('first_edit.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    seen = set()  # To track the first column values we've seen

    for row in csv.reader(inp):
        # Check if we've already seen this value in the first column
        if row[0] not in seen:
            # If not seen, and the second column has at least 1000 characters, write the row
            if len(row[0]) <= 100:
              if len(row[1]) >= 500:
                  writer.writerow(row)
            seen.add(row[0])

print("Rows have been processed and saved to 'first_edit.csv'.")


# Specify your file names
input_file = 'test.csv'
output_file = 'first_edit.csv'

# Define the words to replace and their replacements
replacements = {',  ':','}

def replace_words(text, replacements):
    """
    Replace specified words in the text with given replacements.
    """
    for old_word, new_word in replacements.items():
        text = text.replace(old_word, new_word)
    return text

# Step 2: Process the CSV file
with open(input_file, 'r', newline='', encoding='utf-8') as inp, open(output_file, 'w', newline='', encoding='utf-8') as out:
    # Create CSV reader and writer
    reader = csv.reader(inp)
    writer = csv.writer(out)

    # Process each row
    for row in reader:
        # Replace words in each cell of the row
        new_row = [replace_words(cell, replacements) for cell in row]
        writer.writerow(new_row)


# File names
input_file = 'test.csv'
output_file = 'updated_file.csv'

def filter_rows_with_word(file_name, word):
    """
    Filters out rows containing a specific word, and returns the rows excluding
    the last 200 rows that contain the word.
    """
    filtered_rows = []
    rows_with_word = []

    # Read the CSV file
    with open(file_name, 'r', newline='', encoding='utf-8') as inp:
        reader = csv.reader(inp)
        for row in reader:
            # Check if the row contains the word
            if any(word in cell for cell in row):
                rows_with_word.append(row)
            else:
                filtered_rows.append(row)

    # Exclude the last 200 rows with the word
    if len(rows_with_word) > 20:
        filtered_rows.extend(rows_with_word[:-20])
    else:
        # If fewer than 200 rows contain the word, include all
        filtered_rows.extend(rows_with_word)

    return filtered_rows

# Step 2: Process the file to filter rows
filtered_rows = filter_rows_with_word(input_file, 'Climate ')

with open(output_file, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerows(filtered_rows)

# Download the file
files.download('first_edit.csv')
