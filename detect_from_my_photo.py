from deepface import DeepFace

image_path = "/Users/sabeeha/ED/images/happy.jpeg" # change your image name

result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

print("Emotion detected:", result[0]['dominant_emotion'])
