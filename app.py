# FEELモジュールからFEEL関数をインポート
from FEEL.generator import feel

from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHART_FOLDER'] = 'charts'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHART_FOLDER'], exist_ok=True)

# レーダーチャートを生成
def create_radar_chart(values, filename):
    labels = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
    values = np.append(values, values[0])
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=True)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('Emotion Radar Chart')
    chart_path = os.path.join(app.config['CHART_FOLDER'], filename)
    plt.savefig(chart_path)
    plt.close(fig)
    return chart_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        video = request.files['video']
        if video:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(video_path)
            return render_template('process.html', video=video.filename)
    return render_template('upload.html')

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    video = request.json.get('video')
    if video:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video)
        emotion_vector = feel(video_path)
        chart_filename = f"{video}_radar.png"
        chart_path = create_radar_chart(emotion_vector, chart_filename)
        return jsonify({'chart_url': url_for('chart_file', filename=chart_filename)})
    return jsonify({'error': 'No video provided'}), 400

@app.route('/charts/<filename>')
def chart_file(filename):
    return send_from_directory(app.config['CHART_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
