from deepface import DeepFace
import cv2

# Load your reference images
ref_images = {
    "happy": "/Users/sabeeha/ED/images/happy.jpeg",
    "neutral": "/Users/sabeeha/ED/images/neutral.jpeg",
    "angry":"/Users/sabeeha/ED/images/angry.jpeg",
    "disgust":"/Users/sabeeha/ED/images/disguist.jpeg",
    "surprise":"/Users/sabeeha/ED/images/surprise.jpeg",
}
#stability function---
def stable_emotion(path, repeats=4):
    emotions = []
    for _ in range(repeats):
        try:
            result = DeepFace.analyze(
                img_path=path,
                actions=['emotion'],         # <- correct: 'emotion'
                enforce_detection=False,
            )
            # DeepFace sometimes returns a list [result_dict], handle that:
            if isinstance(result, list) and result:
                res = result[0]
            else:
                res = result

            dom = res.get('dominant_emotion')
            if dom:
                emotions.append(dom)
        except Exception:
            # ignore this attempt (face not found, etc.) and continue
            pass

    # If no successful detection at all, return a safe fallback
    if not emotions:
        return "neutral"

    # return the most frequent detected emotion
    return max(set(emotions), key=emotions.count)

# Pre-analyze your images
ref_results = {}

for emotion, path in ref_images.items():
    ref_results[emotion] = stable_emotion(path)
print("Your Reference Emotions:")
print(ref_results)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        current_emotion = result[0]['dominant_emotion']

        cv2.putText(frame, f'Current Emotion: {current_emotion}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    except:
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Emotion Comparison", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()