import random
import shutil
import os


def create_sample(source_path, target_path, sample_size=15):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    for class_name in os.listdir(source_path):
        class_path = os.path.join(source_path, class_name)
        
        if os.path.isdir(class_path):
            class_dest_path = os.path.join(target_path, class_name)
            if not os.path.exists(class_dest_path):
                os.makedirs(class_dest_path)
            
            images = os.listdir(class_path)
            
            sample_images = random.sample(images, min(sample_size, len(images)))
            
            for image_name in sample_images:
                source_image_path = os.path.join(class_path, image_name)
                dest_image_path = os.path.join(class_dest_path, image_name)
                shutil.copy2(source_image_path, dest_image_path)
