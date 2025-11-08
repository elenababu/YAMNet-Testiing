import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

# Quiet down TF logs 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_MAP_PATH = os.path.join(SCRIPT_DIR, "yamnet_class_map.csv")

# The model files (saved_model.pb, assets/, variables/) are directly in SCRIPT_DIR
YAMNET_MODEL_PATH = SCRIPT_DIR

# Relevant classes mapping
RELEVANT_MAPPING = {
    "Smoke detector, smoke alarm": "Fire alarm",
    "Alarm": "Alarm (generic)",
    "Telephone bell ringing": "Phone ring",
    "Television": "Television",
    "Speech": "Speech",
    "Baby cry, infant cry": "Baby cry",
    "Knock": "Door event",
    "Door, wood door": "Door event",
}


# --------- Audio loading helper ---------
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    waveform, sample_rate = tf.audio.decode_wav(
        file_contents, desired_channels=1
    )
    waveform = tf.squeeze(waveform, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    if sample_rate != 16000:
        waveform = tfio.audio.resample(
            waveform, rate_in=sample_rate, rate_out=16000
        )
    return waveform


# --------- YAMNet class names helper ---------
def get_yamnet_classes():
    import csv

    classes = []
    with open(CLASS_MAP_PATH) as f:
        reader = csv.DictReader(f)  # header: index, mid, display_name
        for row in reader:
            classes.append(row["display_name"])
    return classes


# --------- Main prediction function ---------
def yamnet_predict(filename, min_confidence=0.1):
    print(f"Loading audio: {filename}")
    waveform = load_wav_16k_mono(filename)

    print("Loading YAMNet model from local folder...")
    yamnet_model = hub.load(YAMNET_MODEL_PATH)

    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)

    # Load class names (521 classes)
    classes = get_yamnet_classes()

    # Extract relevant classes only
    category_scores = {}  # e.g. {"Fire alarm": 0.42, "Speech": 0.12, ...}

    for i, class_name in enumerate(classes):
        if class_name in RELEVANT_MAPPING:
            category = RELEVANT_MAPPING[class_name]
            score = float(mean_scores[i])
            # Keep the max score for each category
            if category in category_scores:
                category_scores[category] = max(
                    category_scores[category], score
                )
            else:
                category_scores[category] = score

    if not category_scores:
        print("No relevant classes found at all.")
        return

    # Sort categories by score descending
    sorted_categories = sorted(
        category_scores.items(), key=lambda kv: kv[1], reverse=True
    )

    print("\nRelevant classes only:")
    for category, score in sorted_categories:
        print(f"{category}: {score:.3f}")

    # Show top class with confidence check
    best_category, best_score = sorted_categories[0]
    if best_score < min_confidence:
        print(
            f"\nTop relevant class is {best_category} ({best_score:.3f}), "
            f"but below confidence threshold {min_confidence} → treat as UNKNOWN."
        )
    else:
        print(f"\nTop relevant class: {best_category} ({best_score:.3f})")


if __name__ == "__main__":
    audio_dir = os.path.join(SCRIPT_DIR, "audio")

    # List all files in /audio that end with .wav, .mp3, or .m4a (optional)
    audio_files = [
        f for f in os.listdir(audio_dir)
        if f.lower().endswith((".wav", ".mp3", ".m4a"))
    ]

    print(f"Found {len(audio_files)} audio files in {audio_dir}\n")

    # Loop through and test each file
    for file_name in audio_files:
        audio_path = os.path.join(audio_dir, file_name)
        print("=" * 80)
        print(f"Testing file: {file_name}")
        print("=" * 80)
        try:
            yamnet_predict(audio_path)
        except Exception as e:
            print(f"⚠️  Error processing {file_name}: {e}")
        print("\n")
