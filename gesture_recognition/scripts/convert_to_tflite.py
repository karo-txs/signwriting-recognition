import tensorflow as tf
import os

def convert_to_tflite(saved_model_dir, tflite_model_path):
    tflite_path = os.path.dirname(tflite_model_path)
    os.makedirs(tflite_path, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    converter.optimizations = [] 
    tflite_model = converter.convert()
    
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"Modelo convertido e salvo em: {tflite_model_path}")

saved_model_dir = "assets/experiment_data/dataset_experiment_4/model/fully_connected/best_model"
tflite_model_path = "../models/sw_hagrid_13.tflite"
convert_to_tflite(saved_model_dir, tflite_model_path)
