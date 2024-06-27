import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import cv2
from moviepy.editor import VideoFileClip
from fer import FER
import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import time
import json
import pandas as pd
from generate_reportss import generate_reports
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
import numpy as np
import re
import tempfile
from collections import defaultdict
from pydub import AudioSegment
import os

# Download VADER lexicon
nltk.download('vader_lexicon')
nltk.download('stopwords')

def select_file():
    global video_path
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )
    if video_path:
        file_label.config(text=f"Selected file: {video_path}")
    else:
        file_label.config(text="No file selected")

def proceed():
    if not video_path:
        messagebox.showwarning("Warning", "No file selected. Please select a video file.")
    else:
        print(f"Selected video path: {video_path}")
        root.destroy()

root = tk.Tk()
root.title("Video File Selector")
root.geometry("500x200")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

video_path = ""

select_button = ttk.Button(frame, text="Select Video File", command=select_file)
select_button.grid(column=0, row=0, padx=10, pady=10)

file_label = ttk.Label(frame, text="No file selected")
file_label.grid(column=0, row=1, padx=10, pady=10)

proceed_button = ttk.Button(frame, text="Proceed", command=proceed)
proceed_button.grid(column=0, row=2, padx=10, pady=10)

root.mainloop()

if video_path:
    print(f"Selected video path: {video_path}")
else:
    print("No file selected. Exiting.")

emotion_detector = FER()

def extract_frames(video_path, skip_frames=10):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            frames.append(frame)
        frame_count += 1
    video_capture.release()
    return frames


frames = extract_frames(video_path, skip_frames=10)  # Skip every 10 frames

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions_over_time = []
start_time = time.time()
for frame in frames:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        
        face_image = cv2.resize(face_image, (96, 96))  
        emotions = emotion_detector.detect_emotions(face_image)
        if emotions:
            emotion_label = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            emotions_over_time.append(emotions[0]['emotions'])
    
    cv2.imshow('Video Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

end_time = time.time()
print(f"Processing Time: {end_time - start_time:.2f} seconds")

emotion_aggregate = defaultdict(float)
for emotions in emotions_over_time:
    for emotion, score in emotions.items():
        emotion_aggregate[emotion] += score


total_emotions = sum(emotion_aggregate.values())
for emotion in emotion_aggregate:
    emotion_aggregate[emotion] /= total_emotions

emotion_aggregate = dict(emotion_aggregate)


print("Emotions data saved to emotions.json")

# Save processed frames as a video (optional)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = "C:\\Users\\aishw\\Downloads\\documents\\processed_video.mp4"
out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frames[0].shape[1], frames[0].shape[0]))

for frame in frames:
    out.write(frame)
out.release()

# Extract audio using MoviePy
audio_path = 'output_audio.wav'
clip = VideoFileClip(video_path)
clip.audio.write_audiofile(audio_path)

def convert_audio_to_text(audio_path):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_path)

    chunk_length_ms = 60000  # 1 minute
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    recognizer = sr.Recognizer()
    full_transcript = ""

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(temp_dir, f"chunk{i}.wav")
            chunk.export(chunk_file, format="wav")

            with sr.AudioFile(chunk_file) as source:
                audio_data = recognizer.record(source)
                try:
                    # Use Google Web Speech API (default)
                    text = recognizer.recognize_google(audio_data)
                    full_transcript += text + " "
                except sr.UnknownValueError:
                    print(f"Chunk {i} could not be understood")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Web Speech API; {e}")
                    # Retry after a delay if there is a request error
                    time.sleep(5)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        full_transcript += text + " "
                    except sr.RequestError as e:
                        print(f"Retry failed for chunk {i}; {e}")
                print(f"Processed chunk {i+1}/{len(chunks)}")

    print("Transcribed Text:", full_transcript)
    return full_transcript
def analyze_audio_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)
    total_time = len(y) / sr

    # Extract pitch (using librosa's piptrack)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = float(np.max(pitches))

    # Extract volume (RMS)
    rms = librosa.feature.rms(y=y)
    volume = float(np.mean(rms))

    # Extract speech rate (using tempo)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    speech_rate = float(tempo[0] / 60)  # Convert to words per second

    # Extract pauses and hesitations (silence intervals)
    intervals = librosa.effects.split(y, top_db=20)
    pauses = len(intervals)
    pauses_per_minute = pauses / (total_time * 60)

    print(f"Pitch: {pitch:.2f}")
    print(f"Volume: {volume:.2f}")
    print(f"Speech Rate: {speech_rate:.2f} words/sec")
    print(f"Pauses: {pauses}")
    print(f"Pauses per Minute: {pauses_per_minute:.2f}")

    audio_analysis_results = {
        "pitch": pitch,
        "volume": volume,
        "speech_rate": speech_rate,
        "pauses": pauses,
        "pauses_per_minute": pauses_per_minute,
        "total_time": total_time
    }
    with open('audio_analysis_results.json', 'w') as f:
        json.dump(audio_analysis_results, f, indent=4)

    return total_time, pitch, volume, speech_rate, pauses, pauses_per_minute


#audio_path = 'path_to_your_audio_file.wav'
text = convert_audio_to_text(audio_path)
total_time, pitch, volume, speech_rate, pauses, pauses_per_minute = analyze_audio_features(audio_path)

# Perform sentiment analysis on the transcribed text
def get_sentiment_polarity_subjectivity(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    polarity, subjectivity = get_sentiment_polarity_subjectivity(text)
    sentiment_scores['polarity'] = polarity
    sentiment_scores['subjectivity'] = subjectivity
    return sentiment_scores

# Transcribe audio and analyze sentiment and competencies
transcribed_text = convert_audio_to_text(audio_path)
sentiment_scores = analyze_sentiment(transcribed_text)

# Function to preprocess text for the competency analysis
def preprocess_texts(text):
    stop_words = set(stopwords.words('english'))
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        words = text.split()  # Tokenize
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(words)
    else:
        return ""  # Handle non-string values by returning an empty string

# Load the competency dataset and preprocess it
dataset_path = "C:\\Users\\aishw\\Downloads\\documents\\Video_file\\competency_dataset.csv"
df = pd.read_csv(dataset_path, encoding='latin1')  # Adjust encoding if needed

# Ensure 'text' column has no NaN values
df['text'] = df['text'].fillna("")

# Apply the preprocessing function to the text column
df['Processed_Text'] = df['text'].apply(preprocess_texts)

# Verify preprocessing
print(df[['text', 'Processed_Text']].head())

# Ensure 'competency' column has no NaN values
df['competency'] = df['competency'].fillna("")

# Check data balance
print(df['competency'].value_counts())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Processed_Text'], df['competency'], test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Inspect TF-IDF features
feature_names = tfidf_vectorizer.get_feature_names_out()
print(feature_names[:10])  # Print the first 10 features

# Initialize the classifier
classifier = LogisticRegression()

# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# Evaluate model performance
train_accuracy = classifier.score(X_train_tfidf, y_train)
test_accuracy = classifier.score(X_test_tfidf, y_test)
print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# Predict on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Save the evaluation results to a JSON file
evaluation_results = {
    "accuracy": accuracy,
    "classification_report": classification_rep
}

with open('evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=4)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(json.dumps(classification_rep, indent=4))

# Save the model
joblib.dump(classifier, 'competency_classifier_model.pkl')

# Save the TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Example of loading the model and vectorizer for prediction
# Load the model and vectorizer
loaded_model = joblib.load('competency_classifier_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Process the transcribed text (example)
#transcribed_text = "example transcribed text"
processed_text = preprocess_texts(transcribed_text)
text_tfidf = loaded_vectorizer.transform([processed_text])

# Predict the competency
predicted_competency = loaded_model.predict(text_tfidf)
competency_scores = {predicted_competency[0]: 1}
print(f'Predicted Competency: {predicted_competency[0]}')

# Evaluate candidate
sentiment_threshold = 0.1
emotion_thresholds = {
    "happy": 0.5,
    "neutral": 0.5,
}
competency_threshold = 1



def evaluate_candidate(sentiment_scores, emotions_over_time, competency_scores):
    # Check sentiment
    if sentiment_scores['compound'] < sentiment_threshold:
        return "Rejected due to negative sentiment"

    # Check emotions
    emotion_counts = {key: 0 for key in emotion_thresholds}
    for emotion in emotions_over_time:
        for key in emotion_thresholds:
            if emotion.get(key, 0) > emotion_thresholds[key]:
                emotion_counts[key] += 1
    
    if emotion_counts['happy'] == 0 and emotion_counts['neutral'] == 0:
        return "Rejected due to lack of positive or neutral emotions"
    
    # Check competencies
    for competency, score in competency_scores.items():
        if score < competency_threshold:
            return f"Rejected due to low score in competency: {competency}"
    
    return "Selected"

evaluation_result = evaluate_candidate(sentiment_scores, emotions_over_time, competency_scores)
print(f"Evaluation Result: {evaluation_result}")

# Save results to JSON files
with open('transcribed_text.json', 'w') as f:
    json.dump({"text": transcribed_text}, f)

with open('sentiment_scores.json', 'w') as f:
    json.dump(sentiment_scores, f)

with open('emotion_data.json', 'w') as f:
    json.dump(emotions_over_time, f)

with open('competency_scores.json', 'w') as f:
    json.dump(competency_scores, f)

with open('evaluation_result.json', 'w') as f:
    json.dump({"result": evaluation_result}, f)

with open('emotions.json', 'w') as f:
    json.dump(emotion_aggregate, f)

# Call the report generation

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\b(?:http|https)://\S+\b', '', text)  # Remove URLs
        text = re.sub(r'\W+', ' ', text)  # Remove special characters
        words = text.split()  # Tokenize
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(words)
    else:
        return ""  # Handle non-string values by returning an empty string

def load_and_preprocess_data():
    df = pd.read_csv('mbti_1.csv')
    df['Processed_Posts'] = df['posts'].apply(preprocess_text)
    return df

def train_personality_predictor(df):
    X = df['Processed_Posts']
    y = df['type']
    
    print(y.value_counts())  # Check class distribution

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 1))  # Reduced features
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Handle imbalance by undersampling
    undersampler = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = undersampler.fit_resample(X_train_tfidf, y_train)
    
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }

    random_search = RandomizedSearchCV(LogisticRegression(class_weight='balanced', max_iter=500), param_dist, n_iter=10, cv=3, random_state=42)
    random_search.fit(X_train_res, y_train_res)

    classifier = random_search.best_estimator_
    
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Personality Prediction Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    return classifier, tfidf_vectorizer

def predict_personality(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

# Load and preprocess data
df = load_and_preprocess_data()

# Optionally use a sample for quick testing
# df = df.sample(frac=0.1, random_state=42)  # Use 10% of the data

# Train personality predictor model
personality_model, personality_vectorizer = train_personality_predictor(df)

# Example transcribed text (replace this with actual transcribed text)
#transcribed_text = "Hello, I'm [Name]. I thrive on challenge and achievement, and I'm driven by a vision for success. With a strategic mindset and a knack for leadership, I'm constantly seeking opportunities to innovate and make an impact. I believe in pushing boundaries, setting ambitious goals, and rallying others to achieve greatness together. It's a pleasure to meet you."
# Predict personality from transcribed text
predicted_personality = predict_personality(transcribed_text, personality_model, personality_vectorizer)
print(f'Predicted Personality Type: {predicted_personality}')

# Save the personality prediction result to a JSON file
personality_result = {
    "predicted_personality_accuracy": accuracy,
    "predicted_personality": predicted_personality
}

with open('personality_prediction_result.json', 'w') as json_file:
    json.dump(personality_result, json_file, indent=4)

# Save the personality prediction model and vectorizer
joblib.dump(personality_model, 'personality_predictor_model.pkl')
joblib.dump(personality_vectorizer, 'personality_vectorizer.pkl')

generate_reports()
print("Script execution completed successfully!")
