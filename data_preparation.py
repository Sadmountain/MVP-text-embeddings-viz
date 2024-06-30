import requests
import pandas as pd
from tqdm import tqdm

# Function to fetch authors from OpenAlex for a single DOI
def fetch_authors(doi):
    base_url = 'https://api.openalex.org/works?filter=doi:'
    url = f'{base_url}{doi}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                authors = data['results'][0].get('authorships', [])
                author_names = [author['author']['display_name'] for author in authors]
                return ", ".join(author_names)
            else:
                return 'No authors found'
        else:
            print(f"Failed to fetch data for DOI: {doi}")
            return 'No authors found'
    except Exception as e:
        print(f"Error fetching authors for DOI {doi}: {e}")
        return 'No authors found'

# Read the dataset from a CSV file
data_path = "data/van_de_Schoot_2018.csv"  # Update the path accordingly
papers_df = pd.read_csv(data_path)

# Preprocess DOIs to ensure consistency
papers_df['doi'] = papers_df['doi'].astype(str).str.lower()

# Fetch authors for each paper individually
tqdm.pandas()
papers_df['authors'] = papers_df['doi'].progress_apply(lambda x: fetch_authors(x) if pd.notna(x) else 'No DOI')

# Print summary of authors found
num_no_authors = sum(papers_df['authors'] == 'No authors found')
print(f"Number of papers with no authors found: {num_no_authors}")

# Drop rows with NaN values in the abstract column
papers_df = papers_df.dropna(subset=['abstract'])

# Drop rows with abstracts shorter than 20 words
papers_df = papers_df[papers_df['abstract'].apply(lambda x: len(x.split()) >= 20)]

# Function to preprocess the text (lowercasing, removing punctuation, and tokenizing)
def preprocess(text):
    if text is not None:
        text = text.lower().replace('.', '').replace(',', '').replace(':', '')
        tokens = text.split()
        return ' '.join(tokens)
    return ""

# Apply the preprocessing function to the abstract column
papers_df['processed_abstract'] = papers_df['abstract'].apply(preprocess)

# Save the updated DataFrame to a new CSV file
papers_df.to_csv('data/van_de_Schoot_2018_with_authors.csv', index=False)
print("Data preparation complete. File saved to 'data/van_de_Schoot_2018_with_authors.csv'")
