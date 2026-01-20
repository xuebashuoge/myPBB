import os
import re

def rename_results_folders(root_dir, dry_run=True):
    # Regex Explanation:
    # 1. (.*-chan[\d\.]+-.+?) -> Capture everything up to the end of channel_specs.
    #    It anchors on '-chan<number>-' and stops right before '-kl'.
    # 2. (-kl[\d\.]+.*)       -> Capture the '-kl' parameter and everything following it.
    pattern = re.compile(r"(.*-chan[\d\.]+-.+?)(-kl[\d\.]+.*)")

    print(f"--- Starting Rename Process (Dry Run: {dry_run}) ---")
    
    count = 0
    for dirname in os.listdir(root_dir):
        # Full path handling
        old_path = os.path.join(root_dir, dirname)
        
        # Skip files, only process directories
        if not os.path.isdir(old_path):
            continue

        # Check if folder matches the criteria (contains -chan and -kl)
        # This automatically skips 'objective-vanilla' folders
        match = pattern.match(dirname)
        
        if match:
            # Check if 'frob' is already in the name to prevent double-renaming
            if "-frob-kl" in dirname:
                continue

            # Construct new name: Part1 + "-frob" + Part2
            new_dirname = f"{match.group(1)}-frob{match.group(2)}"
            new_path = os.path.join(root_dir, new_dirname)

            if dry_run:
                print(f"[Preview]\n  OLD: {dirname}\n  NEW: {new_dirname}\n")
            else:
                os.rename(old_path, new_path)
                print(f"[Renamed]\n  FROM: {dirname}\n  TO:   {new_dirname}")
            
            count += 1

    print(f"--- Process Complete. {count} folders processed. ---")

# --- usage ---
# Replace '.' with the actual path to your results parent folder
target_folder = './results/posterior' 

# 1. Run with dry_run=True to verify
rename_results_folders(target_folder, dry_run=False)

# 2. When satisfied, uncomment the line below:
# rename_results_folders(target_folder, dry_run=False)