"""使わない"""
import os
import subprocess
import argparse

def download_kinetics(split, output_dir):
    """
    Download Kinetics-400 dataset based on the given split.
    
    Args:
        split (str): The split to download ('train', 'val', or 'test').
        output_dir (str): The directory to save the downloaded data.
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URL to the list of video IDs for the given split
    url = f"https://raw.githubusercontent.com/activitynet/ActivityNet/master/Crawler/Kinetics/kinetics-400_{split}_list.txt"
    
    # Download the list of video IDs
    subprocess.run(["wget", url, "-O", f"{output_dir}/{split}_list.txt"])
    
    # Read the list of video IDs
    with open(f"{output_dir}/{split}_list.txt", "r") as f:
        video_ids = [line.strip() for line in f]
    
    # Download each video
    for video_id in video_ids:
        output_path = os.path.join(output_dir, f"{video_id}.mp4")
        if not os.path.exists(output_path):
            subprocess.run(["youtube-dl", f"https://www.youtube.com/watch?v={video_id}", "-o", output_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kinetics-400 dataset.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True, help="Directory to save the downloaded data.")
    args = parser.parse_args()

    download_kinetics(args.split, args.output_dir)