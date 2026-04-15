import os
import shutil
import argparse

def flatten_and_rename(src_dir, session_id):
    # The new directory will be named after the session_id
    parent_dir = os.path.dirname(src_dir.rstrip('/\\'))
    dest_dir = os.path.join(parent_dir, session_id)
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
        
    file_count = 0
    
    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # We skip the generated zip if it's there
            if file.endswith('.zip') and 'import' in file:
                continue
                
            file_path = os.path.join(root, file)
            name, ext = os.path.splitext(file)
            
            # Append session_id to the filename
            new_file_name = f"{name}_{session_id}{ext}"
            new_file_path = os.path.join(dest_dir, new_file_name)
            
            # Move the file
            shutil.move(file_path, new_file_path)
            file_count += 1
            
    # Remove the old directory tree
    try:
        shutil.rmtree(src_dir)
        print(f"Removed old directory: {src_dir}")
    except Exception as e:
        print(f"Warning: could not remove {src_dir} completely. {e}")
        
    print(f"Successfully moved and renamed {file_count} files to {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten SVO directory and append Session ID")
    parser.add_argument("--src", default="SVOfile", help="Source directory")
    parser.add_argument("--session", default="OpenArm001", help="Session ID to append and rename folder to")
    
    args = parser.parse_args()
    
    # Resolve absolute paths just to be safe
    base_dir = os.path.abspath(os.path.dirname(__file__))
    src_abs = os.path.join(base_dir, args.src)
    
    if os.path.exists(src_abs):
        flatten_and_rename(src_abs, args.session)
    else:
        print(f"Error: Source directory {src_abs} does not exist.")
