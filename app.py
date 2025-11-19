# app.py

import streamlit as st
import subprocess
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import librosa
import ffmpeg
import io
from IPython.display import display, Markdown
from tqdm import tqdm

# Google Video Intelligence client
from google.oauth2 import service_account
from google.cloud import videointelligence_v1 as videointelligence
from openai import OpenAI
import openai
import anthropic
import google.generativeai as genai

st.set_page_config(page_title="Estimation des effets d'une vidéo YouTube avant publication", layout="wide")
st.title("🎥 Estimation des effets d'une vidéo YouTube avant sa publication")
st.markdown(
    """
    <div style="text-align:center; font-size:18px; margin-top:-10px;">
        <span style="font-size:22px;">📌</span> <strong>Auteurs:</strong><br>
        GBODOGBE Zinsou René | BESSANH Isaac | Crénia
    </div>
    """,
    unsafe_allow_html=True
)
# -----------------------------
# Constants (comme dans ton notebook)
# -----------------------------
VIDEO_DIR = "videos"
AUDIO_DIR = "audios"
GVI_KEY_DIR = "google_video_intelligence"  # dossier où est stockée la clé (comme dans ton notebook)
# exemple dans le notebook: KEY_PATH = "C:/.../google_video_intelligence/xxx.json"
# Ici on cherchera un fichier .json dans ce dossier par défaut, sinon l'utilisateur peut uploader

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(GVI_KEY_DIR, exist_ok=True)

# -----------------------------
# Util: sanitize_filename (fourni par toi)
# -----------------------------
import re

def sanitize_filename(filename: str) -> str:
    """Supprime ou remplace les caractères interdits et espaces."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = re.sub(r"\s+", "_", sanitized)
    return sanitized

# -----------------------------
# Youtube metadata & download (yt-dlp)
# -----------------------------
def download_metadata_yt_dlp(url: str) -> dict:
    """Récupère les métadonnées via yt-dlp -j (json). Retourne dict ou None."""
    try:
        result = subprocess.run(['yt-dlp', '-j', url], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        st.warning(f"Impossible de récupérer metadata via yt-dlp: {e}")
        return None

def download_video_yt_dlp(metadata: dict, out_dir: str = VIDEO_DIR) -> Path:
    """
    Télécharge la vidéo en combinant bestvideo+bestaudio et force mp4 (comme ton notebook).
    Retourne le chemin complet du fichier téléchargé.
    """
    url = metadata.get('webpage_url') or metadata.get('url')
    raw_title = metadata.get('title', f"video_{metadata.get('id','')}")
    safe_title = sanitize_filename(raw_title)
    out_template = os.path.join(out_dir, f"{safe_title}.%(ext)s")
    try:
        subprocess.run([
            'yt-dlp',
            '-f', 'bestvideo+bestaudio',
            '--merge-output-format', 'mp4',
            '-o', out_template,
            url
        ], check=True)
        # on cherche le fichier .mp4 correspondant
        candidates = list(Path(out_dir).glob(f"{safe_title}*.mp4"))
        if candidates:
            # prendre le plus récent
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    except subprocess.CalledProcessError as e:
        st.error(f"Erreur yt-dlp download: {e}")
    return None

# -----------------------------
# Extraction audio & vidéo sans audio (ffmpeg)
# -----------------------------
def extract_audio_and_video_noaudio(input_video_path: str):
    """
    Extrait l'audio en .wav et sauvegarde une vidéo sans audio (_noaudio.mp4).
    Retourne (audio_path, video_noaudio_path)
    """
    p = Path(input_video_path)
    base = p.stem
    audio_out = os.path.join(AUDIO_DIR, f"{base}.wav")
    video_noaudio_out = os.path.join(VIDEO_DIR, f"{base}_noaudio.mp4")

    # === Extraction audio ===
    try:
        (
            ffmpeg
            .input(str(input_video_path))
            .output(audio_out, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:   # <-- remplace ffmpeg.Error par Exception
        st.warning(f"Erreur extraction audio ffmpeg: {e}")

    # === Extraction vidéo sans audio ===
    try:
        (
            ffmpeg
            .input(str(input_video_path))
            .output(video_noaudio_out, vcodec='copy', an=None)
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:   # <-- idem
        st.warning(f"Erreur extraction video noaudio ffmpeg: {e}")

    return audio_out, video_noaudio_out

# -----------------------------
# ffprobe metadata local
# -----------------------------
def get_video_metadata_ffprobe(video_path: str) -> dict:
    """Utilise ffprobe pour récupérer codec, width, height, duration."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_name,codec_type,width,height",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data["format"].get("duration", 0)) if data.get("format") else None
        video_codec, audio_codec, width, height = None, None, None, None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_codec = stream.get("codec_name")
                width = stream.get("width")
                height = stream.get("height")
            elif stream.get("codec_type") == "audio":
                audio_codec = stream.get("codec_name")
        return {"duration": duration, "video_codec": video_codec, "audio_codec": audio_codec, "width": width, "height": height}
    except Exception as e:
        st.warning(f"ffprobe error: {e}")
        return {"duration": None, "video_codec": None, "audio_codec": None, "width": None, "height": None}

# -----------------------------
# Analyse visuelle (adapté de ton notebook)
# -----------------------------
def analyze_visual_quality(video_path: str, sample_frames: int = 30) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"mean_brightness": None, "mean_contrast": None, "mean_sharpness": None, "dominant_color_rgb": None, "dominant_color_hex": None}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return {"mean_brightness": None, "mean_contrast": None, "mean_sharpness": None, "dominant_color_rgb": None, "dominant_color_hex": None}

    sample_indices = np.linspace(0, max(frame_count - 1, 0), min(sample_frames, frame_count)).astype(int)
    brightness, contrast, sharpness, colors = [], [], [], []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(gray.mean())
        contrast.append(gray.std())
        sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        resized = cv2.resize(frame, (100, 100))
        data = resized.reshape(-1, 3)
        dominant_color = tuple(np.round(np.mean(data, axis=0)).astype(int))
        colors.append(dominant_color)
    cap.release()
    colors_np = np.array(colors) if len(colors) > 0 else np.array([])
    mean_color = np.mean(colors_np, axis=0) if colors_np.size > 0 else np.array([0, 0, 0])
    mean_color_hex = '#%02x%02x%02x' % tuple(mean_color.astype(int))
    return {
        "mean_brightness": float(np.mean(brightness)) if brightness else None,
        "mean_contrast": float(np.mean(contrast)) if contrast else None,
        "mean_sharpness": float(np.mean(sharpness)) if sharpness else None,
        "dominant_color_rgb": tuple(mean_color.astype(int)) if colors_np.size > 0 else None,
        "dominant_color_hex": mean_color_hex if colors_np.size > 0 else None
    }

# -----------------------------
# Analyse audio (librosa) (adapté)
# -----------------------------
def analyze_audio_file(audio_path: str) -> dict:
    if not os.path.exists(audio_path):
        return {"rms_volume": None, "spectral_centroid": None, "noise_level": None}
    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        noise_level = float(np.mean(np.abs(y[np.abs(y) < 0.01])) * 100)
        return {"rms_volume": rms, "spectral_centroid": spectral_centroid, "noise_level": noise_level, "audio_sr": sr, "audio_duration": float(librosa.get_duration(y=y, sr=sr))}
    except Exception as e:
        st.warning(f"Librosa erreur: {e}")
        return {"rms_volume": None, "spectral_centroid": None, "noise_level": None}

# -----------------------------
# Google Video Intelligence (adapté de ton notebook)
# -----------------------------
def analyze_video_with_gvi(video_path: str, key_path: str, timeout: int = 900) -> dict:
    """
    Utilise Google Video Intelligence pour SHOT_CHANGE_DETECTION et OBJECT_TRACKING.
    key_path: chemin vers le fichier JSON de service account.
    Retourne dict sommaire (shot_changes_count, object_tracking list simplified).
    """
    try:
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)
        features = [videointelligence.Feature.SHOT_CHANGE_DETECTION, videointelligence.Feature.OBJECT_TRACKING]

        # lecture binaire
        with io.open(video_path, "rb") as f:
            input_content = f.read()
        request = {"input_content": input_content, "features": features}
        operation = client.annotate_video(request=request)
        st.info("Analyse Google Video Intelligence en cours (peut prendre quelques minutes)...")
        result = operation.result(timeout=timeout)
        annotation_result = result.annotation_results[0]

        out = {}
        out["shot_changes_count"] = len(annotation_result.shot_annotations) if annotation_result.shot_annotations else 0
        # object tracking summary
        objects = []
        for obj in (annotation_result.object_annotations or [])[:100]:
            desc = obj.entity.description
            conf = getattr(obj, "confidence", None)
            seg = getattr(obj, "segment", None)
            if seg:
                start = getattr(seg.start_time_offset, "seconds", 0) + getattr(seg.start_time_offset, "microseconds", 0)/1e6
                end = getattr(seg.end_time_offset, "seconds", 0) + getattr(seg.end_time_offset, "microseconds", 0)/1e6
            else:
                start, end = None, None
            objects.append({"description": desc, "confidence": conf, "start_s": start, "end_s": end})
        out["object_tracking"] = objects
        out["object_count"] = len(objects)
        return out
    except Exception as e:
        st.warning(f"GVI erreur: {e}")
        return {"shot_changes_count": None, "object_tracking": None}

# -----------------------------
# Fusion des caractéristiques -> DataFrame
# -----------------------------
def build_feature_dataframe(yt_meta: dict, local_meta: dict, visual: dict, audio: dict, gvi: dict) -> pd.DataFrame:
    """
    Construit une DataFrame avec :
    - Infos principales YouTube : id, titre, description, durée
    - Tous les autres champs (visual, audio, gvi) inchangés
    """
    combined = {}
    combined["video_id"] = yt_meta.get("id")
    combined["Title"] = yt_meta.get("title")
    combined["Description"] = yt_meta.get("description")
    combined["Durations"] = yt_meta.get("duration")
    combined.update(visual)
    combined.update(audio)
    combined.update(gvi)

    return pd.DataFrame([combined])

# -----------------------------
# UI: Sidebar configuration (keys, options)
# -----------------------------
st.sidebar.header("Menu Principal")
sample_frames = st.sidebar.slider("Echantillons d'images", 5, 60, 30)
gvi_use_fixed_key = st.sidebar.checkbox("Utiliser la clé GVI présente dans le dossier google_video_intelligence", value=True)

gvi_key_path_input = None
if not gvi_use_fixed_key:
    gvi_key_path_input = st.sidebar.file_uploader("Uploader la clé JSON GVI (service account)", type=["json"])
else:
    # cherche un .json dans le dossier
    json_candidates = list(Path(GVI_KEY_DIR).glob("*.json"))
    if json_candidates:
        gvi_key_path = str(json_candidates[0])  # utiliser le premier trouvé
    else:
        gvi_key_path = None

openai_key = st.sidebar.text_input("OpenAI API Key (optionnel)", type="password")
anthropic_key = st.sidebar.text_input("Anthropic API Key (optionnel)", type="password")
gemini_key = st.sidebar.text_input("Gemini API Key (optionnel)", type="password")

st.sidebar.markdown("**Notes**: la clé GVI est lue depuis le dossier `google_video_intelligence/` si la case est cochée, sinon vous pouvez uploader un .json.")

# -----------------------------
# Tabs (Style A)
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📥 Téléchargement", "🧪 Extraction & Caractéristiques", "🧠 Analyse LLM"])

# State placeholders
if "current_video" not in st.session_state:
    st.session_state["current_video"] = None
if "yt_metadata" not in st.session_state:
    st.session_state["yt_metadata"] = None
if "features_df" not in st.session_state:
    st.session_state["features_df"] = None

# -----------------------------
# TAB 1: Téléchargement
# -----------------------------
with tab1:
    st.header("📥 Téléchargement / Importation")

    source = st.radio("Source de la vidéo", ("YouTube URL", "Upload local"), index=0)

    if source == "YouTube URL":
        yt_url = st.text_input("Collez le lien YouTube (ex: https://www.youtube.com/watch?v=...)")
        if st.button("Récupérer métadonnées YouTube"):
            if not yt_url:
                st.error("Entrez une URL YouTube valide.")
            else:
                meta = download_metadata_yt_dlp(yt_url)
                if meta:
                    st.session_state["yt_metadata"] = meta
                    st.success("Métadonnées récupérées.")
                    st.json({k: meta.get(k) for k in ("id", "title", "description", "duration", "view_count", "like_count", "channel")})
                else:
                    st.error("Impossible de récupérer les métadonnées.")
        if st.button("Télécharger la vidéo depuis YouTube"):
            if st.session_state.get("yt_metadata") is None:
                st.error("Récupérez d'abord les métadonnées (bouton précédent).")
            else:
                with st.spinner("Téléchargement via yt-dlp..."):
                    video_path = download_video_yt_dlp(st.session_state["yt_metadata"], out_dir=VIDEO_DIR)
                    if video_path:
                        st.session_state["current_video"] = str(video_path)
                        st.success(f"Téléchargé: {video_path.name}")
                        st.video(str(video_path))
                    else:
                        st.error("Erreur téléchargement.")
    else:
        uploaded = st.file_uploader("Téléversez une vidéo (mp4, webm, mov, mkv ...)", type=["mp4","webm","mov","mkv","avi"])
        if uploaded:
            # sauvegarder avec nom nettoyé
            safe_name = sanitize_filename(uploaded.name)
            dest_path = os.path.join(VIDEO_DIR, safe_name)
            with open(dest_path, "wb") as f:
                f.write(uploaded.read())
            st.session_state["current_video"] = dest_path
            st.success(f"Fichier sauvegardé: {dest_path}")
            st.video(dest_path)

    # Afficher informations si vidéo courante
    if st.session_state.get("current_video"):
        st.markdown(f"**Fichier actif:** `{st.session_state['current_video']}`")
        if st.session_state.get("yt_metadata"):
            st.markdown("**Métadonnées YouTube (si disponibles):**")
            st.json({k: st.session_state['yt_metadata'].get(k) for k in ("id","title","description","duration","view_count","like_count","channel")})

# -----------------------------
# TAB 2: Extraction & Caractéristiques
# -----------------------------
with tab2:
    st.header("🧪 Extraction & Caractéristiques")

    st.write("Cliquez pour extraire audio, metadata locaux, métriques visuelles, Google Video Intelligence, puis construire la DataFrame.")

    if st.button("Extraire caractéristiques complètes"):
        cur = st.session_state.get("current_video")
        if not cur:
            st.error("Aucune vidéo active. Importez ou téléchargez une vidéo dans l'onglet précédent.")
        else:
            st.info("Extraction : metadata local (ffprobe), audio, visual quality, GVI .")
            with st.spinner("FFprobe (métadonnées locales)..."):
                local_meta = get_video_metadata_ffprobe(cur)

            with st.spinner("Extraction audio & vidéo sans audio (ffmpeg)..."):
                audio_path, video_noaudio_path = extract_audio_and_video_noaudio(cur)
                st.session_state["last_audio"] = audio_path
                st.session_state["last_noaudio"] = video_noaudio_path

            with st.spinner("Analyse audio (librosa)..."):
                audio_feats = analyze_audio_file(audio_path)

            with st.spinner("Analyse visuelle (frames)..."):
                visual_feats = analyze_visual_quality(cur, sample_frames=sample_frames)

            # GVI: déterminer chemin cle
            gvi_results = {}
            gvi_key_to_use = None
            if gvi_use_fixed_key:
                # si on a une json dans GVI_KEY_DIR
                if 'gvi_key_path' in locals() and gvi_key_path:
                    gvi_key_to_use = gvi_key_path
            else:
                # si user upload la clé
                if gvi_key_path_input:
                    # sauvegarder temporairement la clé
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                    tmp.write(gvi_key_path_input.getvalue())
                    tmp.flush()
                    tmp.close()
                    gvi_key_to_use = tmp.name

            if gvi_key_to_use:
                with st.spinner("Analyse Google Video Intelligence (peut prendre du temps)..."):
                    gvi_results = analyze_video_with_gvi(cur, gvi_key_to_use)
            else:
                st.info("Aucune clé Google Video Intelligence trouvée — GVI ignoré. Placez le JSON dans le dossier google_video_intelligence/ ou uploadez ci-dessus.")

            # YouTube metadata si existante
            yt_meta = st.session_state.get("yt_metadata", {})
            df = build_feature_dataframe(yt_meta, local_meta, visual_feats, audio_feats, gvi_results)
            st.session_state["features_df"] = df
            st.subheader("DataFrame des caractéristiques")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger (.csv)", csv, file_name=f"features_{Path(cur).stem}.csv")

# -----------------------------
# TAB 3: Analyse LLM
# -----------------------------
with tab3:
    st.header("🧠 Analyse avec LLMs (ChatGPT, Claude, Gemini)")

    if st.session_state.get("features_df") is None:
        st.warning("Aucune DataFrame disponible — exécutez l'extraction dans l'onglet précédent.")
    else:
        df = st.session_state["features_df"]
        st.subheader("Caractéristiques utilisées pour l'analyse")
        st.dataframe(df)

        llm_choice = st.selectbox("Choisir un LLM pour l'analyse", ("Local summary (fallback)", "ChatGPT (OpenAI)", "Claude (Anthropic)", "Gemini (Google)"))
        user_prompt = st.text_area("Prompt (optionnel) — instruction donnée au modèle:", value="You are an expert in video marketing. Analyze this video based on its characteristics and provide scores (0-100) and recommendations per aspect. Provide a final average score and three concrete recommendations to improve performance before publishing.")

        if st.button("Lancer l'analyse LLM"):
            features_json = df.to_json(orient="records", force_ascii=False)
            prompt = f"{user_prompt}\n\nVideo characteristics (json):\n{features_json}"

            if llm_choice == "Local summary (fallback)":
                # résumé local simple (heuristique)
                r = df.iloc[0].to_dict()
                lines = []
                lines.append(f"Durée (s): {r.get('duration') or r.get('yt_duration') or r.get('audio_duration')}")
                lines.append(f"Résolution: {r.get('width')}x{r.get('height')}")
                if r.get('mean_sharpness'):
                    bl = r['mean_sharpness']
                    lines.append(f"Netteté (variance Laplacian moyenne): {bl:.1f} — {'Net' if bl>200 else 'Potentiellement flou'}")
                if r.get('rms_volume'):
                    lines.append(f"Audio RMS: {r.get('rms_volume'):.4f}")
                if r.get('object_count'):
                    lines.append(f"Objets détectés (GVI) : {r.get('object_count')}")
                lines.append("Recommandations: vérifier l'éclairage si brightness < 60; augmenter le niveau audio si RMS faible.")
                st.subheader("Résumé local")
                st.write("\n".join(lines))
            elif llm_choice == "ChatGPT (OpenAI)":
                if not openai_key:
                    st.error("Fournir une clé OpenAI dans la barre latérale pour utiliser ChatGPT.")
                else:
                    try:
                        import openai
                        client = OpenAI(api_key=openai_key)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system",
                                 "content": "You are an expert assistant in video marketing analysis.."},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.7,
                        )
                        # Extracting plain text from the template
                        output_text = response.choices[0].message.content
                        st.markdown(output_text)
                    except Exception as e:
                        st.error(f"Erreur appel OpenAI: {e}")
            elif llm_choice == "Claude (Anthropic)":
                if not anthropic_key:
                    st.error("Fournir une clé Anthropic (Claude) dans la barre latérale pour utiliser Claude.")
                else:
                    try:
                        from anthropic import Anthropic
                        client = anthropic.Client(api_key=anthropic_key)
                        # exemple simple
                        prompt_claude = f"{prompt}"
                        resp = client.messages.create(model="claude-sonnet-4-20250514",max_tokens=1000,
                                                  messages=[
                                                      {"role": "user", "content": prompt_claude}]
                                                     )
                        output_text = resp.content[0].text
                        st.subheader("Réponse Claude")
                        st.markdown(output_text)
                    except Exception as e:
                        st.error(f"Erreur appel Claude: {e}")
            else:  # Gemini
                if not gemini_key:
                    st.error("Fournir une clé Gemini dans la barre latérale (ou configurez genai).")
                else:
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=gemini_key)
                        model = "gemini-flash-latest"  # adapter si nécessaire
                        res = genai.GenerativeModel(model).generate_content(prompt)
                        st.subheader("Réponse Gemini")
                        st.markdown(res.text)
                    except Exception as e:
                        st.error(f"Erreur appel Gemini: {e}")

# -----------------------------
# Footer: aide & dépendances
# -----------------------------
st.markdown("---")
