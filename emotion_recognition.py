from deepface import DeepFace
import cv2

def recognize_emotion(image_path):
    img = cv2.imread(image_path)
    result = DeepFace.analyze(img, actions=['emotion'])
    emotions = result['emotion']
    
    return emotions

if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'  
    emotions = recognize_emotion(image_path)
    
    print("Emotion probabilities:")
    print(emotions)
