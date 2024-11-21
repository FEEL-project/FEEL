import os
from pathlib import Path
import subprocess

# EmVidCapの場合
# 動画ファイルが保存されているディレクトリ
# SOURCE_DIR = "/home/u01231/project_body/FEEL/data/EmVidCap/Videos/EmVidCap-L/TrainVal_clips/Test"
# 分割後のファイルを保存するディレクトリ
# OUTPUT_DIR = "/home/u01231/project_body/FEEL/data/EmVidCap/Videos/EmVidCap-L/TrainVal_clips/splitted_Test"

# joeの場合
SOURCE_DIR = "/home/u01231/project_body/FEEL/data/youtube_movies/joe"
OUTPUT_DIR = "/home/u01231/project_body/FEEL/data/youtube_movies/joe/splitted"

os.makedirs(OUTPUT_DIR, exist_ok=True)  # 出力ディレクトリを作成

# 動画の長さを取得する関数
def get_video_duration(file_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return float(result.stdout.strip())
    return 0.0

# 動画を10秒ごとに分割する関数
def split_video(input_file, output_dir, segment_duration, start_index):
    output_files = []
    segment_index = start_index

    # 動画の長さを取得
    video_duration = get_video_duration(input_file)

    # 分割コマンド実行
    for start_time in range(0, int(video_duration), segment_duration):
        # 各セグメントの出力パス
        output_path = os.path.join(output_dir, f"{segment_index:04d}.mp4")
        
        # 動画を切り取るためのffmpegコマンド
        command = [
            "ffmpeg",
            "-ss", str(start_time),  # 開始時間を指定
            "-i", input_file,
            "-c:v", "libx264",  # エンコードを強制
            "-c:a", "aac",      # 音声エンコード（必要なら）
            "-strict", "experimental",
            "-t", str(segment_duration),  # セグメントの長さを指定
            "-f", "mp4",         # 出力形式
            output_path
        ]
        
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # セグメントの長さをチェックし、3秒未満なら削除
        duration = get_video_duration(output_path)
        if duration < 3.0:  # 3秒未満なら削除
            print(f"Deleting short segment: {output_path} (duration: {duration:.2f}s)")
            os.remove(output_path)  # ファイル削除
        else:
            output_files.append(output_path)
            segment_index += 1

    return output_files, segment_index

# 動画処理ループ
video_files = sorted(Path(SOURCE_DIR).glob("**/*.mp4"))  # サブディレクトリも含む
start_index = 1

for video_file in video_files:
    try:
        print(f"Processing: {video_file}")
        _, start_index = split_video(str(video_file), OUTPUT_DIR, segment_duration=10, start_index=start_index)
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {video_file}: {e}")

print("Processing complete.")
