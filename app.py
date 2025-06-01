from flask import Flask, request, send_file, render_template, render_template_string
import os
import librosa
import librosa.display
import numpy as np
np.float = float
np.int = int
import matplotlib.pyplot as plt
import uuid
import zipfile
import shutil

app = Flask(__name__)
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('audio')
    if not file:
        return 'No file uploaded.', 400

    original_filename = file.filename
    base_name = os.path.splitext(original_filename)[0]
    session_id = str(uuid.uuid4())
    working_dir = os.path.join(TMP_DIR, session_id)
    os.makedirs(working_dir, exist_ok=True)

    audio_path = os.path.join(working_dir, original_filename)
    file.save(audio_path)

    # Load audio (90 premières secondes uniquement)
    y, sr = librosa.load(audio_path, sr=None, duration=90)

    # Extract features
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Analyse de tonalité avancée (majeur/minor)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma_cqt.mean(axis=1)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    correlations = []
    for i in range(12):
        corr_major = np.corrcoef(np.roll(major_profile, i), chroma_mean)[0, 1]
        corr_minor = np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0, 1]
        correlations.append((corr_major, f"{notes[i]} major"))
        correlations.append((corr_minor, f"{notes[i]} minor"))
    best_match = max(correlations, key=lambda x: x[0])
    key_name = best_match[1]

    # Suite des features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    energy = np.mean(librosa.feature.rms(y=y)) * 100
    danceability = np.std(librosa.onset.onset_strength(y=y, sr=sr)) * 10
    aggressiveness = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) / 1000
    loudness = np.mean(librosa.feature.rms(y=y))
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_percussive_ratio = np.mean(harmonic) / (np.mean(percussive) + 1e-6)
    tonal_complexity = np.var(chroma_cqt) * 100
    beat_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr)) * 10
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Définitions simples pour l'annexe
    explanations = {
        "BPM": "Battements par minute : indique la vitesse du morceau.",
        "Key": "Ton musical principal (note) du morceau.",
        "Energy": "Niveau d'énergie global, plus c'est élevé, plus le morceau est dynamique.",
        "Danceability": "Facilité avec laquelle on peut danser sur le morceau.",
        "Aggressiveness": "Sensibilité aux sons agressifs ou percutants.",
        "Loudness": "Volume moyen perçu du morceau.",
        "Harmonic/Percussive Ratio": "Rapport entre parties harmoniques (mélodie) et percussives (rythme).",
        "Tonal Complexity": "Complexité tonale, ou diversité des notes et accords.",
        "Beat Strength": "Force et présence du rythme.",
        "Spectral Centroid": "Indique si le son est plus aigu ou grave."
    }

    # Export TXT avec annexe explicative
    txt_path = os.path.join(working_dir, 'audio-features.txt')
    with open(txt_path, 'w') as f:
        f.write(f"AUDIO ANALYSIS - {original_filename}\n\n")
        bpm_val = float(bpm) if not isinstance(bpm, float) else bpm
        f.write(f"BPM: {bpm_val:.2f}\n")
        f.write(f"Key: {key_name}\n")
        f.write(f"Energy: {energy:.2f}%\n")
        f.write(f"Danceability: {danceability:.2f}\n")
        f.write(f"Aggressiveness: {aggressiveness:.2f}\n")
        f.write(f"Loudness: {loudness:.6f}\n")
        f.write(f"Harmonic/Percussive Ratio: {harmonic_percussive_ratio:.4f}\n")
        f.write(f"Tonal Complexity: {tonal_complexity:.2f}\n")
        f.write(f"Beat Strength: {beat_strength:.2f}\n")
        f.write(f"Spectral Centroid: {spectral_centroid:.2f}\n")

        # Annexe explicative
        f.write("\n\n---\n")
        f.write("ANNEXE: Explications des audio-features\n\n")
        for key, desc in explanations.items():
            f.write(f"{key} : {desc}\n")

    # Graphs
    viz_path = os.path.join(working_dir, 'visualizations')
    os.makedirs(viz_path, exist_ok=True)

    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, 'spectrogram.png'))
    plt.close()

    # Chromagram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, 'chromagram.png'))
    plt.close()

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_path, 'mfcc.png'))
    plt.close()

    # Export ZIP
    zip_filename = f"{base_name}_{session_id}.zip"
    zip_path = os.path.join(TMP_DIR, zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(working_dir):
            for f in files:
                if f.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                    continue  # Ignore le fichier audio original
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, working_dir)
                zipf.write(abs_path, arcname=rel_path)

    # Nettoyage du dossier temporaire (on garde le zip)
    shutil.rmtree(working_dir)

    # Retourne une petite page HTML à l'iframe
    return render_template_string(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Upload Complete</title></head>
        <body>
            <script>
                // Signal vers le parent
                window.parent.postMessage("upload-complete", "*");
                // Lien de téléchargement direct
                window.location.href = "/download/{zip_filename}";
            </script>
        </body>
        </html>
    """)

@app.route('/download/<filename>')
def download(filename):
    zip_path = os.path.join(TMP_DIR, filename)
    if not os.path.exists(zip_path):
        return "File not found", 404

    try:
        return send_file(zip_path, as_attachment=True)
    finally:
        try:
            os.remove(zip_path)
        except:
            pass

if __name__ == '__main__':
    app.run(debug=True)
