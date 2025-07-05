#!/usr/bin/env python3
"""
Script to rename RGBT dataset files by removing hash values
This ensures proper pairing between visible and infrared images for YOLO-MIF RGBT training
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

def extract_base_name(filename):
    """
    Extract the base name from a filename with hash
    Examples:
    - BIRD_00160_001_png.rf.83d82e1339a851b2e3f37bc0631284d8.jpg -> BIRD_00160_001_png
    - IR_BIRD_00160_001_png.rf.69a4850f48f928d16a41dd24ecb8f454.jpg -> IR_BIRD_00160_001_png
    """
    # Remove the .rf.{hash}.{extension} part
    match = re.match(r'(.+?)\.rf\.[a-f0-9]{32}\.(jpg|txt)', filename)
    if match:
        return match.group(1)
    return filename.split('.')[0]  # fallback

def create_sequential_mapping(files, is_infrared=False):
    """
    Create a mapping from original filenames to sequential numbered filenames
    For infrared files, remove IR_ prefix to match visible counterparts
    """
    # Group files by their base name (without hash)
    base_groups = defaultdict(list)
    for filename in files:
        base_name = extract_base_name(filename)
        # Remove IR_ prefix for grouping - this ensures IR and RGB files get same numbers
        clean_base = base_name.replace('IR_', '')
        base_groups[clean_base].append(filename)
    
    # Create mapping with sequential numbering
    mapping = {}
    counter = 1
    
    for clean_base, file_list in sorted(base_groups.items()):
        for i, filename in enumerate(sorted(file_list)):
            # Create new filename: BIRD_001, BIRD_002, etc.
            # Both visible and infrared will have the same naming pattern (no IR_ prefix)
            if len(file_list) > 1:
                new_name = f"BIRD_{counter:03d}_{chr(97+i)}"  # a, b, c for multiple versions
            else:
                new_name = f"BIRD_{counter:03d}"
            
            mapping[filename] = new_name
        counter += 1
    
    return mapping

def rename_files_in_directory(directory_path, file_mapping, file_extension):
    """
    Rename files in a directory based on the mapping
    """
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return
    
    renamed_count = 0
    for old_filename, new_base_name in file_mapping.items():
        old_path = directory / old_filename
        new_filename = f"{new_base_name}.{file_extension}"
        new_path = directory / new_filename
        
        if old_path.exists():
            try:
                # Rename the file
                old_path.rename(new_path)
                print(f"Renamed: {old_filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {old_filename}: {e}")
        else:
            print(f"File not found: {old_filename}")
    
    print(f"Renamed {renamed_count} files in {directory_path}")

def main():
    # Base path to your train_data folder
    base_path = Path("train_data")
    
    if not base_path.exists():
        print(f"Error: {base_path} directory not found!")
        print("Please run this script from the project root directory.")
        return
    
    print("🔄 Starting dataset file renaming process...")
    print("=" * 60)
    
    # Define all directories to process
    directories = [
        "images/visible/train",
        "images/visible/val", 
        "images/infrared/train",
        "images/infrared/val",
        "labels/visible/train",
        "labels/visible/val",
        "labels/infrared/train", 
        "labels/infrared/val"
    ]
    
    # Process each directory
    for dir_path in directories:
        full_path = base_path / dir_path
        if not full_path.exists():
            print(f"⚠️  Directory not found: {full_path}")
            continue
            
        print(f"\n📁 Processing: {dir_path}")
        
        # Get list of files
        files = [f.name for f in full_path.iterdir() if f.is_file()]
        if not files:
            print(f"   No files found in {dir_path}")
            continue
            
        print(f"   Found {len(files)} files")
        
        # Determine file extension
        if "images" in dir_path:
            extension = "jpg"
        else:  # labels
            extension = "txt"
            
        # Check if this is an infrared directory
        is_infrared = "infrared" in dir_path
        
        # Create mapping (both visible and infrared will have same naming pattern)
        file_mapping = create_sequential_mapping(files, is_infrared)
        
        # Rename files
        rename_files_in_directory(full_path, file_mapping, extension)
    
    print("\n" + "=" * 60)
    print("✅ Dataset renaming completed!")
    print("\n📋 Next steps:")
    print("1. Verify that visible and infrared images now have IDENTICAL names")
    print("   - visible/train/BIRD_001.jpg should match infrared/train/BIRD_001.jpg")
    print("   - Same for label files")
    print("2. Run your RGBT training script")
    print("3. The YAML file should work perfectly with matching names")

if __name__ == "__main__":
    main()
