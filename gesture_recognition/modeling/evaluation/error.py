from base.hand_gesture import landmark_collection, landmark_normalization, landmark_visualization
from tensorflow.keras.models import load_model
from base.base.utilities import config
from collections import defaultdict
import tensorflow as tf
import argparse
import os
import json


def load_and_preprocess_image(output_dir, image_path, save_images=False):
    """Carrega e processa uma imagem para extração de landmarks da mão."""
    landmarks = landmark_collection.get_highest_hand_landmark_data_from_path(
        image_path, save_landmark_image=False)
    if not landmarks:
        return None

    image_name = os.path.basename(image_path)
    if save_images:
        # Salva a imagem de landmarks original
        landmark_visualization.draw_landmarks_on_image(
            [landmarks["hand_landmark"]
             ], f"{output_dir}/landmarks/{image_name}"
        )

    # Normalização dos landmarks
    landmarks['hand_landmark'] = landmark_normalization.apply_all_normalizations(
        landmarks['hand_landmark'])
    landmarks['world_hand'] = landmark_normalization.apply_all_normalizations(
        landmarks['world_hand'])

    if save_images:
        # Salva a imagem de landmarks normalizada
        landmark_visualization.draw_landmarks_on_image(
            [landmarks["hand_landmark"]
             ], f"{output_dir}/landmarks_normalized/{image_name}"
        )

    return landmarks['hand_landmark'], tf.constant(landmarks['handedness']), landmarks['world_hand']


def get_labels_map(dataset_dir):
    """Gera um mapa de rótulos baseado na estrutura do diretório de dataset."""
    label_map = {}
    for idx, class_name in enumerate(sorted(os.listdir(dataset_dir))):
        if os.path.isdir(os.path.join(dataset_dir, class_name)):
            label_map[idx] = class_name
    return label_map


def get_key(label_dict, value):
    """Recupera a chave de um dicionário dado um valor."""
    for key, val in label_dict.items():
        if val == value:
            return key


def get_image_paths_and_labels(root_dir, label_map):
    """Coleta os caminhos das imagens e seus rótulos verdadeiros."""
    file_paths, true_labels = [], []

    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    file_paths.append(os.path.join(class_dir, file_name))
                    true_labels.append(get_key(label_map, class_name))

    return file_paths, true_labels


def infer_and_save_results(model, image_paths, true_labels, label_map, output_dir):
    """Realiza inferência e salva os resultados em arquivos específicos."""
    correct_path = os.path.join(output_dir, 'correct_classifications.txt')
    incorrect_path = os.path.join(output_dir, 'incorrect_classifications.txt')
    no_hand_path = os.path.join(output_dir, 'no_hand_detected.txt')
    os.makedirs(os.path.join(output_dir, "landmarks"), exist_ok=True)
    os.makedirs(os.path.join(
        output_dir, "landmarks_normalized"), exist_ok=True)

    with open(correct_path, 'a') as correct_file, \
            open(incorrect_path, 'a') as incorrect_file, \
            open(no_hand_path, 'a') as no_hand_file:

        for i, image_path in enumerate(image_paths):
            # Carrega e processa a imagem, salvando landmarks e landmarks normalizados
            result = load_and_preprocess_image(
                output_dir, image_path, save_images=True)
            if result is None:
                no_hand_file.write(f"{image_path}\n")
                print(f"[WARNING] No hands detected for {image_path}.")
                continue

            landmarks, handedness, world_hand = result
            prediction = model.predict([tf.expand_dims(landmarks, 0),
                                        tf.expand_dims(handedness, 0),
                                        tf.expand_dims(world_hand, 0)])
            predicted_label = tf.argmax(prediction, axis=-1).numpy()[0]
            true_label = true_labels[i]

            # Salva o caminho da imagem nos arquivos corretos e incorretos
            if predicted_label == true_label:
                correct_file.write(f"{image_path}\n")
            else:
                incorrect_file.write(
                    f"{image_path} (Pred: {label_map[predicted_label]}, True: {label_map[true_label]})\n")

    print(f"[INFO] Inference complete. Results saved in {output_dir}")


def generate_error_report(output_dir, json_output_file='error_report.json'):
    """Gera um relatório de erros de classificação e salva em um arquivo JSON."""
    incorrect_classifications_path = os.path.join(
        output_dir, 'incorrect_classifications.txt')
    error_report = defaultdict(lambda: defaultdict(int))

    with open(incorrect_classifications_path, 'r') as incorrect_file:
        for line in incorrect_file:
            line = line.strip()
            if line:
                true_class = line.split('True: ')[-1].replace(")", "").strip()
                predicted_class = line.split(
                    'Pred: ')[-1].split(',')[0].strip()
                error_report[true_class][predicted_class] += 1

    for true_class, predicted_classes in error_report.items():
        total_errors = sum(predicted_classes.values())
        for predicted_class in predicted_classes:
            error_report[true_class][predicted_class] = round(
                (predicted_classes[predicted_class] / total_errors) * 100, 2
            )

    json_output_path = os.path.join(output_dir, json_output_file)
    with open(json_output_path, 'w') as json_file:
        json.dump(error_report, json_file, indent=4)

    print(f"[INFO] Error report saved to {json_output_path}")


def error_analysis(base_data_path, package_cfg, experiment_path, training_cfg):
    for dataset in package_cfg:
        if dataset.get("use") == "test":
            source_path = f"{base_data_path}/{dataset.get('dataset')}/images"
            output_result_path = f"{experiment_path}/evaluate/{training_cfg.get('model')}/error_analysis"
            model_path = f"{experiment_path}/model/{training_cfg.get('model')}/best_model"

            label_map = get_labels_map(source_path)
            image_paths, true_labels = get_image_paths_and_labels(
                source_path, label_map)

            model = load_model(model_path)

            infer_and_save_results(
                model, image_paths, true_labels, label_map, output_result_path)
            generate_error_report(output_result_path)
