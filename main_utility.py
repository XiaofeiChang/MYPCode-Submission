import os
import shutil

def clean_all_subfolders(parent_folder):
    '''
    Clear all the images in subfolders for initialization.
    :return: 
    '''''
    for subdir in os.listdir(parent_folder):
        subdir_path = os.path.join(parent_folder, subdir)
        if os.path.isdir(subdir_path):
            if os.listdir(subdir_path):
                # print(f"{subdir} is not empty. Deleting files inside...")
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                # print("Files deleted successfully.")
            # Check if subfolder is empty
            # else:
                # print(f"{subdir} is empty.")
        # else:
        #     print(f"{subdir} is not a subfolder.")