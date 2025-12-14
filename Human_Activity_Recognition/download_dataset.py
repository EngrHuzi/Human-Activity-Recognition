"""
Script to download and extract the UCI HAR Dataset
"""
import urllib.request
import zipfile
import os
import sys

def download_dataset():
    """Download and extract the UCI HAR dataset"""

    dataset_url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
    zip_file_name = 'human_activity_recognition_using_smartphones.zip'

    # Check if dataset already exists
    if os.path.exists('UCI HAR Dataset'):
        print("✓ Dataset already exists!")
        return True

    print("Downloading UCI HAR Dataset...")
    print(f"URL: {dataset_url}")
    print("This may take a few minutes (dataset is ~60MB)...")

    try:
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            sys.stdout.write(f'\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)')
            sys.stdout.flush()

        urllib.request.urlretrieve(dataset_url, zip_file_name, show_progress)
        print("\n✓ Download complete!")

        # Extract first zip
        print("\nExtracting archive...")
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("✓ First extraction complete!")

        # Extract second zip (UCI HAR Dataset.zip)
        if os.path.exists('UCI HAR Dataset.zip'):
            print("Extracting UCI HAR Dataset...")
            with zipfile.ZipFile('UCI HAR Dataset.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            print("✓ Second extraction complete!")

            # Clean up
            os.remove('UCI HAR Dataset.zip')

        # Remove downloaded zip
        os.remove(zip_file_name)
        print("✓ Cleanup complete!")

        # Verify extraction
        if os.path.exists('UCI HAR Dataset'):
            print("\n" + "="*50)
            print("✓ SUCCESS! Dataset downloaded and extracted!")
            print("="*50)
            print("\nDataset location: UCI HAR Dataset/")
            print("\nYou can now:")
            print("1. Refresh your browser at http://localhost:8000")
            print("2. Click 'Load Dataset' button")
            print("3. Start training models!")
            return True
        else:
            print("\n✗ Error: Dataset folder not found after extraction")
            return False

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print("https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones")
        return False

if __name__ == "__main__":
    download_dataset()
