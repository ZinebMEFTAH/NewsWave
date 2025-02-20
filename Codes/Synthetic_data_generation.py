# set environment variables
# https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = "..."

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator, OPENAI_TEMPLATE
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_SUFFIX, SYNTHETIC_FEW_SHOT_PREFIX

class NewsHeadlines(BaseModel):
    Keywords: str
    Articles: str

import csv

# Initialize the list to store the examples
examples = []

# Manually define the column names based on their order in the file
column_names = ['Keywords', 'Articles']


# Open the CSV file and read its contents
with open('seed.csv', mode='r') as file:
    # Create a CSV reader without headers
    csv_reader = csv.reader(file)

    # Iterate through each row in the CSV file
    for row in csv_reader:
        # Map the row data to the column names
        row_data = dict(zip(column_names, row))

        # Format the data into the required string
        example_string = f"""The news article's keywords: {row_data['Keywords']}, The news article: {row_data['Articles']}"""

        # Append the formatted string as a dictionary to the examples list
        examples.append({"example": example_string})

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

custom_prompt = f"""
Each time create a new set of keywords then generate its news article.
"""

# Create a FewShotPromptTemplate with the custom prompt
prompt_template = FewShotPromptTemplate(
    prefix=custom_prompt,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)

synthetic_data_generator = create_openai_data_generator(
    output_schema=NewsHeadlines,
    llm=ChatOpenAI(model="ft:gpt-4o-mini-2024-07-18:open-ai-website:test5:9w9ELVAc", temperature=1),
    prompt=prompt_template,
)

synthetic_results = synthetic_data_generator.generate(
    subject="News Articles",
    extra="The keywords and news articles should be varied, diversed, and not repeated.",
    runs=100,
)


import pandas as pd
synthetic_datas =[]
# Create a list of dictionaries from the objects
for item in synthetic_results:
    synthetic_datas.append({
        'Keywords': item.Keywords,
        'Articles': item.Articles
    })
#synthetic_df = []
# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_datas)


from google.colab import files

csv_file_path = 'synthetic_data.csv'
synthetic_df.to_csv(csv_file_path, index=False)

# If you're using Google Colab, you can download the file directly
files.download(csv_file_path)
