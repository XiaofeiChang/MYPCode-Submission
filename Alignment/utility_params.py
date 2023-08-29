'''
This file is to construct the params.json file. The params.json is used for parameter settings.
'''''
import json
import os



class Params():
    '''
    Class that loads hyperparameters from params.json file.
    '''''
    def __init__(self, json_path):
        with open(json_path) as f:
            # Load the file
            params = json.load(f)
            self.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, dict):
        # Loads parameters from json file
        self.__dict__.update(dict)

    @property
    def dict(self):
        # Gives dict-like access to Params instance
        return self.__dict__


def get_params_path():
    '''
    Get the path of params.json
    :return: str: path
    '''''
    # Get the current directory (where the utility_params.py is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the Alignment path
    substring_to_remove = "/Alignment"
    # Get the upper directory, which is the directory of the project
    proj_dir = current_dir.replace(substring_to_remove, "")
    # Construct the path of params.json
    params_path = proj_dir + '/params.json'

    return params_path