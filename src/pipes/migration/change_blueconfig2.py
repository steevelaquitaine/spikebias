"""Rename the original path to the new path where the simulation was moved in the blueconfig file of each simulation of a campaign

Used to migrate raw Blue brain simulations to another path after bbp_wrokflow
"""

import os
import os
import re

# campaign 01
original_path1 = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/neuropixels_lfp_10m_384ch_hex01_rou04_pfr03_40Khz_2024_11_16_disconnected/3d803ad9-1e0d-4f7b-a892-5a28fca7a536/"
destination_path1 = "/gpfs/bbp.cscs.ch/project/proj85/laquitai/preprint_2024/raw/neuropixels/npx_384ch_hex01_rou04_pfr03_40Khz_2024_11_16_disconnected/3d803ad9-1e0d-4f7b-a892-5a28fca7a536/"

# campaign 02
original_path2 = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/neuropixels_lfp_10m_384ch_hex01_rou04_pfr03_40Khz_2024_11_16_disconnected_campaign2/6196211a-f976-4610-b3da-701903e378a8/"
destination_path2 = "/gpfs/bbp.cscs.ch/project/proj85/laquitai/preprint_2024/raw/neuropixels/npx_384ch_hex01_rou04_pfr03_40Khz_2024_11_16_disconnected_campaign2/6196211a-f976-4610-b3da-701903e378a8/"

# campaign 03
original_path3 = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/neuropixels_lfp_10m_384ch_hex01_rou04_pfr03_40Khz_2024_11_24_disconnected_campaign3/da3af241-0200-46af-be9e-49ada91a1d26"
destination_path3 = "/gpfs/bbp.cscs.ch/project/proj85/laquitai/preprint_2024/raw/neuropixels/npx_384ch_hex01_rou04_pfr03_40Khz_2024_11_24_disconnected_campaign3/da3af241-0200-46af-be9e-49ada91a1d26/"


def find_and_replace_in_files(directory, find_text, replace_text, file_extensions=None):
    """
    Perform a find and replace in all files within a directory and its subdirectories.

    Args:
        directory (str): The path to the root directory where the search will start.
        find_text (str): The text pattern to search for.
        replace_text (str): The text to replace the search pattern with.
        file_extensions (list): List of file extensions to consider. If None, all files are included.
    """
    # Compile the find pattern for faster replacements
    find_pattern = re.compile(re.escape(find_text))
    # print(find_text)
    
    # Walk through each file in the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        
        for file_name in files:
            
            # Check if the file extension matches if specified
            if file_extensions and not any(file_name.endswith(ext) for ext in file_extensions):
                continue
           
            file_path = os.path.join(root, file_name)
            
            # Read the file content
            with open(file_path, 'r') as file:
                content = file.read()
                
            # Replace all occurrences of the find pattern with the replacement text
            new_content = find_pattern.sub(replace_text, content)
            
            # Write the updated content back to the file if changes were made
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"Updated: {file_path}")



find_and_replace_in_files(destination_path1, original_path1, destination_path1, file_extensions=[".json", "BlueConfig"])

find_and_replace_in_files(destination_path2, original_path2, destination_path2, file_extensions=[".json", "BlueConfig"])

find_and_replace_in_files(destination_path3, original_path3, destination_path3, file_extensions=[".json", "BlueConfig"])