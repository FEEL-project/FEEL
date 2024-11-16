"""
downloader.shを実行して取得した動画データを frame ごとの画像データに変換する

要件: `sudo apt install ffmpeg` が実行済みであること
"""
import os
import subprocess
import argparse

# 動画が保存されたフォルダ「kinetics_videos」にある、クラスの種類とパスを取得
# dir_path = './data/kinetics_videos'

def convert_mp4_to_jpeg(dir_path):
    class_list = os.listdir(path=dir_path)
    print(class_list)

    # 各クラスの動画ファイルを画像ファイルに変換する
    for class_list_i in (class_list):  # クラスごとのループ

        # クラスのフォルダへのパスを取得
        class_path = os.path.join(dir_path, class_list_i)

        # 各クラスのフォルダ内の動画ファイルをひとつずつ処理するループ
        for file_name in os.listdir(class_path):

            # ファイル名と拡張子に分割
            name, ext = os.path.splitext(file_name)

            # mp4ファイルでない、フォルダなどは処理しない
            if ext != '.mp4':
                continue

            # 動画ファイルを画像に分割して保存するフォルダ名を取得
            dst_directory_path = os.path.join(class_path, name)

            # 上記の画像保存フォルダがなければ作成
            if not os.path.exists(dst_directory_path):
                os.mkdir(dst_directory_path)

            # 動画ファイルへのパスを取得
            video_file_path = os.path.join(class_path, file_name)

            # ffmpegを実行させ、動画ファイルをjpgにする （高さは256ピクセルで幅はアスペクト比を変えない）
            # kineticsの動画の場合10秒になっており、大体300ファイルになる（30 frames /sec）
            cmd = 'ffmpeg -i \"{}\" -vf scale=-1:256 \"{}/image_%05d.jpg\"'.format(
                video_file_path, dst_directory_path)
            print(cmd)
            subprocess.call(cmd, shell=True)
            print('\n')

    print("動画ファイルを画像ファイルに変換しました。")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mp4 video to jpeg image.")
    parser.add_argument('--data-dir', '-d', required=True, 
                        default='kinetics-400/train', 
                        help="Directory to the data formatted ../data/\{path to dir\}.")
    args = parser.parse_args()

    convert_mp4_to_jpeg('../data/' + args.data_dir)