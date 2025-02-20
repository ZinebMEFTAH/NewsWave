pip install requests
pip install pandas requests
import requests

# Define your API key
api_key = '...'

# Define the endpoint URL
url = 'https://api.openai.com/v1/chat/completions'

# Define the headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}

# Define the data payload
data = {
    "model": "gpt-4",  # Specify the model version you are using
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. I will provide you with Keywords and you should generate a news article about them."},
        {"role": "user", "content": "What's the weather like today?"}
    ]
}

# Make the POST request
response = requests.post(url, headers=headers, json=data)

# Print the response
print(response.json())


import pandas as pd
import requests
import json

# Define the API endpoint and your access token
api_url = 'https://api.openai.com/v1/chat/completions'
api_key = '...'  # Replace with your actual API key

# Define the input and output CSV files
input_csv_file = 'input.csv'  # CSV file with tags in the first column
output_csv_file = 'responses.csv'  # CSV file to save the generated articles

# Function to send a message to the OpenAI API
def get_response(user_message):
    payload = {
        "model": "ft:gpt-4o-mini-2024-07-18:open-ai-website:test5:9w9ELVAc",  # Specify the model version you are using
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. I will provide you with Keywords and you should generate a news article about them. Do not start always writing the article using the first keyword, try instead to make it editorialist."},
            {"role": "user", "content": user_message}
        ]
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    response_json = response.json()
    # Extract the assistant's reply from the response
    return response_json.get('choices', [{}])[0].get('message', {}).get('content', '')

# Read the input CSV file
df = pd.read_csv(input_csv_file)

# Initialize a list to store the responses
responses = []


for index, row in df.iterrows():
    # Access columns by name
    user_message = row.iloc[0]
    response = get_response(user_message)
    responses.append({'Keywords': user_message, 'Articles': response})


# Create a DataFrame from the responses and save to a new CSV file
responses_df = pd.DataFrame(responses)
responses_df.to_csv(output_csv_file, index=False)

print(f'Responses have been saved to {output_csv_file}')

from google.colab import files


# If you're using Google Colab, you can download the file directly
files.download(output_csv_file)
