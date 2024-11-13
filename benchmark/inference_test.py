import tensorflow as tf
import numpy as np
import time
import argparse
import math

def load_tflite_model(model_path):
    with open(model_path, "rb") as f:
        tflite_model = f.read()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    return interpreter

def generate_random_inputs():
    node_features = np.random.rand(1, 21, 3).astype(np.float32)
    handness_input = np.random.rand(1, 1).astype(np.float32)
    landmarks_word_input = np.random.rand(1, 21, 3).astype(np.float32)
    return node_features, handness_input, landmarks_word_input

def run_inference(interpreter, n_inferences=100):
    input_details = interpreter.get_input_details()
    times = []
    for _ in range(n_inferences):
        node_features, handness_input, landmarks_word_input = generate_random_inputs()
        interpreter.set_tensor(input_details[0]['index'], node_features)
        interpreter.set_tensor(input_details[1]['index'], handness_input)
        interpreter.set_tensor(input_details[2]['index'], landmarks_word_input)

        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        
        times.append(end_time - start_time)
    return times

def calculate_statistics(times):
    mean_time = np.mean(times)
    std_dev = np.std(times, ddof=1)
    ci_95 = 1.96 * (std_dev / math.sqrt(len(times)))
    throughput = 1 / mean_time if mean_time > 0 else 0
    return mean_time, std_dev, ci_95, throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste de inferência com TensorFlow Lite")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho para o modelo TFLite")
    parser.add_argument("--n_inferences", type=int, default=100, help="Número de inferências para calcular o tempo médio")
    parser.add_argument("--repeat", type=int, default=10, help="Número de repetições para cada configuração")
    args = parser.parse_args()

    interpreter = load_tflite_model(args.model_path)

    all_times = []
    for _ in range(args.repeat):
        times = run_inference(interpreter, args.n_inferences)
        all_times.extend(times)
    
    mean_time, std_dev, ci_95, throughput = calculate_statistics(all_times)

    # Formatação para evitar notação científica
    print(f"{mean_time:.5f},{std_dev:.5f},{ci_95:.5f},{throughput:.2f}")
