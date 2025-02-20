# Paths to the input JSONL file and output CSV file
jsonl_file_path = 'input.jsonl'
csv_file_path = 'output.csv'

# Read existing rows if the CSV file already exists
existing_rows = []
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    existing_rows = list(reader)

# Open the CSV file for writing (append mode)
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonlfile:
        existing_rows[0].append("Articles")
        for index, line in enumerate(jsonlfile):
            if index == 0:
                continue
            json_data = json.loads(line)

            # Access the 'choices' list within the 'response'->'body'
            choices = json_data.get('response', {}).get('body', {}).get('choices', [])

            # Find the first message with the role 'assistant'
            assistant_content = None
            for choice in choices:
                message = choice.get('message', {})
                if message.get('role') == 'assistant':
                    assistant_content = message.get('content')
                    break

            existing_rows[index].append(assistant_content)

    writer.writerows(existing_rows)


print("CSV file created and updated successfully!")
files.download(csv_file_path)
