import os
from datasets import load_dataset
from PIL import Image

def save_fer2013():
    print("Downloading FER-2013 from HuggingFace...")
    dataset = load_dataset("AutumnQiu/fer2013")
    
    # FER-2013 emotions mapping
    emotions = {
        0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
        4: 'sad', 5: 'surprise', 6: 'neutral'
    }
    
    base_dir = "data/fer2013"
    
    # Create directories
    for split in ['train', 'test']:
        for emotion in emotions.values():
            os.makedirs(os.path.join(base_dir, split, emotion), exist_ok=True)
            
    print("Saving Training images...")
    for idx, item in enumerate(dataset['train']):
        img = item['image']
        label = item['label']
        if label in emotions:
            img.save(os.path.join(base_dir, 'train', emotions[label], f"train_{idx}.png"))
            
    print("Saving Testing images...")
    for idx, item in enumerate(dataset['test']):
        img = item['image']
        label = item['label']
        if label in emotions:
            img.save(os.path.join(base_dir, 'test', emotions[label], f"test_{idx}.png"))

    print("Dataset successfully downloaded and organized!")

if __name__ == "__main__":
    save_fer2013()
