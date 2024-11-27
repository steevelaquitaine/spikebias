import os
import subprocess

#ji_simulation_root = "/gpfs/bbp.cscs.ch/project/proj83/home/isbister/simulations/"
#physiology_paper_root = ji_simulation_root + "elife_sscx_physiology_2024/"

physiology_paper_root = "/gpfs/bbp.cscs.ch/project/proj85/laquitai/preprint_2024/raw/neuropixels/npx_384ch_hex01_rou04_pfr03_40Khz_2024_11_16_disconnected"


# directory_pairs = {

#     "0-UnconnectedScan-Original": {
#         "original_path": "/gpfs/bbp.cscs.ch/project/proj83/scratch/bbp_workflow/sscx_O1_conductance_callibration",
#         "nexus_path": "https://bbp.epfl.ch/nexus/v1/resources/bbp/somatosensorycortex/_/9da0abcb-25fe-43a4-8d5b-6cc3402fc3e4",
#         "destination_path": physiology_paper_root + "0-UnconnectedScan/0-UnconnectedScan-Original",
#         "move": False,
#         "change_paths": True
#     }

# }


directory_pairs = {

    "0-UnconnectedScan-Original": {
        "original_path": "/gpfs/bbp.cscs.ch/project/proj83/scratch/bbp_workflow/sscx_O1_conductance_callibration",
        "nexus_path": "https://bbp.epfl.ch/nexus/v1/resources/bbp/somatosensorycortex/_/9da0abcb-25fe-43a4-8d5b-6cc3402fc3e4",
        "destination_path": physiology_paper_root + "0-UnconnectedScan/0-UnconnectedScan-Original",
        "move": False,
        "change_paths": True
    }

}


import os
import re
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
        # print(root)
        for file_name in files:
            # Check if the file extension matches if specified
            if file_extensions and not any(file_name.endswith(ext) for ext in file_extensions):
                continue
           
            # print(file_name) 
            file_path = os.path.join(root, file_name)
            # print(file_path)
            
            # Read the file content
            with open(file_path, 'r') as file:
                content = file.read()

                # print(content)

            # print(replace_text)
                
            # Replace all occurrences of the find pattern with the replacement text
            new_content = find_pattern.sub(replace_text, content)
            # print(new_content)
            
            # Write the updated content back to the file if changes were made
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"Updated: {file_path}")

for key, opts in directory_pairs.items():
	if opts["change_paths"]: 
		print(key)
		find_and_replace_in_files(opts["destination_path"], opts["original_path"], opts["destination_path"], file_extensions=[".json", "BlueConfig"])
		# find_and_replace_in_files("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/physiology_2023/cortexetl/configs", opts["original_path"], opts["destination_path"], file_extensions=[".yaml"])
		# find_and_replace_in_files("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/simulations/spike_sorting/", "/gpfs/bbp.cscs.ch/project/proj68/scratch/tharayil/newCoeffs/b3122c45-e2d9-4205-821c-88f28791dc69/0/neuropixels_full_O1/", "/gpfs/bbp.cscs.ch/project/proj83/home/laquitai/spike_sorting_electrode_weights_nov2024/", file_extensions=["BlueConfig"])


        


print_paths = False
if print_paths:
    for key, opts in directory_pairs.items():
        print(opts['destination_path'])


# def move_bbp_workflow_simulation_campaign(key, opts):

# 	source = opts["nexus_path"]
# 	target = opts["destination_path"]

# 	print(key)

# 	# Create target directory if it doesn't exist
# 	os.makedirs(target, exist_ok=True)

# 	# Execute the bbp-workflow launch command
# 	command = [
# 	    "bbp-workflow", "launch", "--follow", "bbp_workflow.simulation",
# 	    "MoveSimCampaign", f"sim-config-url={source}", f"path-prefix={target}"
# 	]
# 	try:
# 	    subprocess.run(command, check=False)
# 	except:
# 	    print("next")

# 	# Print status message
# 	print(f"Moved {source} to {target} ({key})")


# """
# Press ctrl-c after each one is moved to move on to the next
# """
# for key, opts in directory_pairs.items():
# 	if opts['move']: move_bbp_workflow_simulation_campaign(key, opts)

# print("All copy operations completed.")