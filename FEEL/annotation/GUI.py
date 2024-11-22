import gradio as gr
import pandas as pd
from pathlib import Path
import os

# 保存先のデータベース（CSVファイル）の初期化
# EmVidCapの場合
# DATABASE_FILE = "../params_trainval.csv"
# VIDEO_DIR = "../data/EmVidCap/Videos/EmVidCap-L/TrainVal_clips/splitted_TrainVal"

# joeの場合
# DATABASE_FILE = "/home/u01231/project_body/FEEL/annotation/joe/params_trainval.csv"
# VIDEO_DIR = "/home/u01231/project_body/FEEL/data/youtube_movies/joe/splitted/trainval"
DATABASE_FILE = "/home/u01231/project_body/FEEL/annotation/joe/params_test.csv"
VIDEO_DIR = "/home/u01231/project_body/FEEL/data/youtube_movies/joe/splitted/test"

if not Path(DATABASE_FILE).exists():
    pd.DataFrame(columns=["video", "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]).to_csv(
        DATABASE_FILE, index=False
    )

# 動画ディレクトリと動画リストの準備
video_dir_path = Path(VIDEO_DIR)
video_files = sorted(video_dir_path.glob("*.mp4"))  # 動画ファイルリスト
# print(f"Number of videos: {len(video_files)}, #1: {video_files[0]}")


# 未処理の動画を取得する関数
def get_next_video():
    # 保存済みデータベースのロード
    if Path(DATABASE_FILE).exists():
        saved_data = pd.read_csv(DATABASE_FILE)
        processed_videos = saved_data["video"].tolist()
    else:
        processed_videos = []
    
    # 未処理の動画をリストアップ
    for video_file in video_files:
        if os.path.basename(str(video_file)) not in processed_videos:
            return str(video_file)
    return None


# データベースにデータを保存する関数
def save_parameters(video_path, joy, trust, fear, surprise, sadness, disgust, anger, anticipation):
    # データを保存
    data = {
        "video": [os.path.basename(video_path)],
        "joy": [joy],
        "trust": [trust],
        "fear": [fear],
        "surprise": [surprise],
        "sadness": [sadness],
        "disgust": [disgust],
        "anger": [anger],
        "anticipation": [anticipation],
    }
    df = pd.DataFrame(data)
    if Path(DATABASE_FILE).exists():
        df_existing = pd.read_csv(DATABASE_FILE)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(DATABASE_FILE, index=False)

    # 次の動画を準備
    next_video_path = get_next_video()
    if next_video_path is None:
        return None, None, "Parameters saved successfully! No more videos to process."
    return next_video_path, next_video_path, "Parameters saved successfully!"


# 初期動画を取得
initial_video_path = get_next_video()


# GUI構築
with gr.Blocks(css=".slider { margin: 5px 0 !important; }") as demo:
    gr.Markdown("### Annotate Videos with Parameters")
    
    with gr.Row():
        with gr.Column(scale=2):
            # 動画表示エリア
            video_player = gr.Video(value=initial_video_path, label="Video Player")
            video_path_display = gr.Textbox(value=initial_video_path, label="Current Video", interactive=False)
        with gr.Column(scale=1):
            # パラメータスライダー
            params = [
                gr.Slider(0, 1, step=0.1, value=0, label=emotion)
                for emotion in ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
            ]

    # 保存ボタン
    save_button = gr.Button("Save Parameters")
    # 出力エリア
    output_message = gr.Textbox(label="Message", interactive=False)

    # ボタンのクリックイベント
    save_button.click(
        fn=save_parameters,
        inputs=[video_path_display] + params,
        outputs=[video_player, video_path_display, output_message],
    )

# GUIを起動
demo.launch()
