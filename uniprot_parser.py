from io import TextIOWrapper
import pandas as pd
import requests

def transform_entry(entry: dict):
    sequence: str = entry['sequence']['value']
    features = ['C'] * len(sequence)

    for feature in entry['features']:
        value = 'C'

        if feature['type'] == 'Helix':
            value = 'H'
        elif feature['type'] == 'Beta strand':
            value = 'B'
        else:
            continue

        for j in range(feature['location']['start']['value'] - 1, feature['location']['end']['value']):
            features[j] = value
    
    features = ''.join(features)
    
    return sequence, features

def collect_to_file(URL: str, n: int, filepath: str):
    data: dict = {'sequence': [], 'structure': []}

    for page in range(n):
        response: requests.Response = requests.get(URL)

        link_header: str = response.headers['Link']
        start: int = link_header.find('<') + 1
        end: int = link_header.find('>')
        
        URL = link_header[start:end]

        response_json: list[dict] = response.json()['results']

        for entry in response_json:
            sequence, features = transform_entry(entry)

            if (sequence.find('X') > -1):
                continue

            data['sequence'].append(sequence)
            data['structure'].append(features)
        
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(filepath)

    return URL

def get_uniprot_data():
    URL: str = "https://rest.uniprot.org/uniprotkb/search?format=json&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28proteins_with%3A1%29&size=500&sort=id+asc"

    URL = collect_to_file(URL, 1, 'data/training.csv')
    URL = collect_to_file(URL, 1, 'data/test.csv')    

if __name__ == "__main__":
    get_uniprot_data()