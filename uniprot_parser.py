'''Module for functions relating to the Uniprot API'''
import pandas as pd
import requests

def transform_entry(entry: dict) -> tuple[str, str]:
    '''
    Read the given JSON entry for a protein, and returns the protein and structure sequence.

    Parameters:
        entry (dict): The JSON entry for a protein from Uniprot
    
    Returns:
        sequence (str): the protein sequence
        structure (str): a string defining the secondary structure.
                        C for coil, H for helix, and B for Beta strand
    '''
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

        feature_start: int = feature['location']['start']['value'] - 1
        feature_end: int = feature['location']['end']['value']

        for j in range(feature_start, feature_end):
            features[j] = value

    features = ''.join(features)

    return sequence, features

def collect_to_file(url: str, n: int, filepath: str) -> str:
    '''
    Will call the Uniprot API n times and store the data to the given filepath.
    The response is processed so that only protein sequences 
    and a custom structure sequence is stored.

    Parameters:
        url (str): The URL of the API to call.
        n (int): The number of times to call the API for this file.
        filepath: (str): The file path to store the csv to.
    
    Returns:
        str: The next URL to call.
    '''
    data: dict = {'sequence': [], 'structure': []}

    for _page in range(n):
        response: requests.Response = requests.get(url, timeout=10.0)

        link_header: str = response.headers['Link']
        start: int = link_header.find('<') + 1
        end: int = link_header.find('>')

        url = link_header[start:end]

        response_json: list[dict] = response.json()['results']

        for entry in response_json:
            sequence, features = transform_entry(entry)

            if sequence.find('X') > -1:
                continue

            data['sequence'].append(sequence)
            data['structure'].append(features)

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(filepath)

    return url

def get_uniprot_data() -> None:
    '''
    Calls the uniprot API and processes the data, to then store it in csv files in the 'data' dir.
    '''
    url: str = ("https://rest.uniprot.org/uniprotkb/search?format=json"
                "&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28proteins_with%3A1%29"
                "&size=500&sort=id+asc")

    url = collect_to_file(url, 8, 'data/training.csv')
    url = collect_to_file(url, 2, 'data/test.csv')

if __name__ == "__main__":
    get_uniprot_data()
