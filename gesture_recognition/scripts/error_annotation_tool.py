import os
import csv
import random
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Caminhos
REPORT_FILE = "assets/experiment_data/tests_experiment_1/evaluate/fully_connected/simple/report.csv"
OUTPUT_DIR = "assets/experiment_data/tests_experiment_1/evaluate/fully_connected/error_analysis"
INCORRECT_FILE = os.path.join(OUTPUT_DIR, "incorrect_classifications.txt")
CORRECT_FILE = os.path.join(OUTPUT_DIR, "correct_classifications.txt")
NO_HAND_FILE = os.path.join(OUTPUT_DIR, "no_hand_detected.txt")
SAVE_FILE = os.path.join(OUTPUT_DIR, "error_annotations.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "error_summary.json")

# Função para ler a acurácia original do arquivo CSV de relatório


def get_original_accuracy():
    """Lê a acurácia original do modelo a partir de report.csv."""
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            report_data = next(reader)
            return float(report_data["accuracy"])
    return None


# Leitura do arquivo de classificações incorretas
incorrect_samples = []
with open(INCORRECT_FILE, 'r') as f:
    for line in f:
        if line.strip():
            image_path = line.split()[0]
            class_name = os.path.basename(os.path.dirname(image_path))
            incorrect_samples.append((class_name, image_path))

# Configuração do CSV de saída
with open(SAVE_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["nome_imagem", "classe", "tipo_erro"])

# Função para incluir as entradas de `no_hand_detected.txt` no CSV


def add_no_hand_detected_entries():
    """Adiciona entradas do arquivo no_hand_detected.txt ao CSV como `mediapipe_not_found`."""
    if os.path.exists(NO_HAND_FILE):
        with open(NO_HAND_FILE, 'r') as f:
            no_hand_entries = f.readlines()
        with open(SAVE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for entry in no_hand_entries:
                entry = entry.strip()
                if entry:
                    # Obtém apenas o nome do arquivo
                    file_name = os.path.basename(entry)
                    class_name = os.path.basename(os.path.dirname(entry))
                    writer.writerow(
                        [file_name, class_name, "mediapipe_not_found"])


# Adiciona entradas de no_hand_detected.txt ao CSV
add_no_hand_detected_entries()

# Função para carregar amostras corretas da mesma classe


def load_correct_examples(class_name, exclude_image, n=3):
    correct_samples = []
    with open(CORRECT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and os.path.basename(os.path.dirname(line)) == class_name:
                correct_samples.append(line)

    examples = random.sample(
        [img for img in correct_samples if img != exclude_image], min(len(correct_samples), n))
    return examples


def count_all_correct_examples():
    correct_samples = []
    with open(CORRECT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            correct_samples.append(line)
    return len(correct_samples)

# Classe para a interface de anotação


class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Error Annotation Tool")
        self.root.configure(bg="white")  # Define o fundo como branco
        self.root.geometry("1400x900")  # Aumenta o tamanho da janela

        self.index = 0
        # Total de amostras para análise
        self.total_samples = len(incorrect_samples)

        self.original_accuracy = get_original_accuracy()  # Acurácia original do modelo
        self.create_widgets()
        self.load_sample()

    def create_widgets(self):
        # Exibição do progresso no canto superior esquerdo
        self.progress_label = tk.Label(
            self.root, text="", font=("Arial", 12), bg="white")
        self.progress_label.place(x=10, y=10)

        # Frame da Amostra Incorreta (lado esquerdo)
        incorrect_frame = tk.Frame(self.root, bg="white")
        incorrect_frame.place(x=20, y=50)

        incorrect_title = tk.Label(incorrect_frame, text="Misclassified Sample", font=(
            "Arial", 14, "bold"), bg="white")
        incorrect_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.incorrect_image_label = tk.Label(incorrect_frame, bg="white")
        self.incorrect_image_label.grid(row=1, column=0, padx=10, pady=5)

        self.incorrect_landmark_label = tk.Label(incorrect_frame, bg="white")
        self.incorrect_landmark_label.grid(row=2, column=0, padx=10, pady=5)

        self.incorrect_normalized_label = tk.Label(incorrect_frame, bg="white")
        self.incorrect_normalized_label.grid(row=3, column=0, padx=10, pady=5)

        # Linha de separação vertical
        separator = tk.Frame(self.root, width=2, height=700, bg="gray")
        separator.place(x=400, y=50)

        # Frame dos Exemplos Corretos (lado direito)
        correct_frame = tk.Frame(self.root, bg="white")
        correct_frame.place(x=450, y=50)

        correct_title = tk.Label(correct_frame, text="Correctly Classified Examples", font=(
            "Arial", 14, "bold"), bg="white")
        correct_title.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        self.example_frame = tk.Frame(correct_frame, bg="white")
        self.example_frame.grid(row=1, column=0, columnspan=3)

        # Frame dos Botões de Anotação (meio inferior)
        self.error_buttons_frame = tk.Frame(self.root, bg="white")
        self.error_buttons_frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

        self.mediapipe_button = tk.Button(
            self.error_buttons_frame, text="MediaPipe Error", bg="#FF9999", command=lambda: self.save_annotation("mediapipe")
        )
        self.mediapipe_button.grid(row=0, column=0, padx=10, pady=10)

        self.model_button = tk.Button(
            self.error_buttons_frame, text="Model Error", bg="#ADD8E6", command=lambda: self.save_annotation("model")
        )
        self.model_button.grid(row=0, column=1, padx=10, pady=10)

    def load_sample(self):
        # Atualiza o progresso
        self.progress_label.config(
            text=f"Progress: {self.index + 1} of {self.total_samples} samples reviewed"
        )

        # Verifica se já passou por todas as amostras
        if self.index >= self.total_samples:
            self.save_summary()  # Salva o resumo dos resultados ao final
            messagebox.showinfo("Info", "Annotation completed!")
            self.root.quit()
            return

        # Obtém a classe e o caminho da imagem incorreta
        class_name, image_path = incorrect_samples[self.index]
        image_name = os.path.basename(image_path)

        # Define a imagem e a classe atuais antes de verificar o landmark
        self.current_image = image_name
        self.current_class = class_name

        # Caminhos dos landmarks
        incorrect_landmark_path = os.path.join(
            OUTPUT_DIR, "landmarks", image_name)
        incorrect_normalized_path = os.path.join(
            OUTPUT_DIR, "landmarks_normalized", image_name)

        # Exibe a imagem incorreta
        self.incorrect_image = Image.open(image_path)
        self.incorrect_image.thumbnail((250, 250))
        self.tk_incorrect_image = ImageTk.PhotoImage(self.incorrect_image)
        self.incorrect_image_label.config(image=self.tk_incorrect_image)

        # Exibe o landmark da imagem incorreta
        self.incorrect_landmark = Image.open(incorrect_landmark_path)
        self.incorrect_landmark.thumbnail((250, 250))
        self.tk_incorrect_landmark = ImageTk.PhotoImage(
            self.incorrect_landmark)
        self.incorrect_landmark_label.config(image=self.tk_incorrect_landmark)

        # Exibe o landmark normalizado da imagem incorreta
        self.incorrect_normalized = Image.open(incorrect_normalized_path)
        self.incorrect_normalized.thumbnail((250, 250))
        self.tk_incorrect_normalized = ImageTk.PhotoImage(
            self.incorrect_normalized)
        self.incorrect_normalized_label.config(
            image=self.tk_incorrect_normalized)

        # Limpa exemplos corretos antigos e exibe novos
        for widget in self.example_frame.winfo_children():
            widget.destroy()

        # Exibe a classe da amostra incorreta
        correct_examples = load_correct_examples(class_name, image_path)
        for i, example_path in enumerate(correct_examples):
            # Exibe imagem correta
            example_image = Image.open(example_path)
            example_image.thumbnail((100, 100))
            tk_example_image = ImageTk.PhotoImage(example_image)
            example_label = tk.Label(
                self.example_frame, image=tk_example_image, bg="white")
            example_label.image = tk_example_image
            example_label.grid(row=0, column=i, padx=5, pady=5)

            # Exibe o landmark da imagem correta
            example_landmark_path = os.path.join(
                OUTPUT_DIR, "landmarks", os.path.basename(example_path))
            example_landmark = Image.open(example_landmark_path)
            example_landmark.thumbnail((100, 100))
            tk_example_landmark = ImageTk.PhotoImage(example_landmark)
            example_landmark_label = tk.Label(
                self.example_frame, image=tk_example_landmark, bg="white")
            example_landmark_label.image = tk_example_landmark
            example_landmark_label.grid(row=1, column=i, padx=5, pady=5)

            # Exibe o landmark normalizado da imagem correta
            example_normalized_path = os.path.join(
                OUTPUT_DIR, "landmarks_normalized", os.path.basename(example_path))
            example_normalized = Image.open(example_normalized_path)
            example_normalized.thumbnail((100, 100))
            tk_example_normalized = ImageTk.PhotoImage(example_normalized)
            example_normalized_label = tk.Label(
                self.example_frame, image=tk_example_normalized, bg="white")
            example_normalized_label.image = tk_example_normalized
            example_normalized_label.grid(row=2, column=i, padx=5, pady=5)

    def save_annotation(self, error_type):
        # Salva a anotação no CSV
        with open(SAVE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [self.current_image, self.current_class, error_type])

        # Passa para a próxima amostra
        self.index += 1
        self.load_sample()

    def save_summary(self):
        """Calcula e salva o resumo dos erros em um arquivo JSON."""
        counts = {"model_error": 0, "mediapipe_error": 0,
                  "mediapipe_not_found_error": 0}

        # Lê o arquivo de anotações e conta cada tipo de erro
        with open(SAVE_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Pula o cabeçalho
            for row in reader:
                if row[2] == "model":
                    counts["model_error"] += 1
                elif row[2] == "mediapipe":
                    counts["mediapipe_error"] += 1
                elif row[2] == "mediapipe_not_found":
                    counts["mediapipe_not_found_error"] += 1

        counts["total_error"] = sum(counts.values())
        counts["total_error_without_not_found"] = counts["total_error"] - \
            counts["mediapipe_not_found_error"]
        counts["total_samples"] = len(
            incorrect_samples) + count_all_correct_examples()

        # Calcula a acurácia ajustada removendo mediapipe_error
        total_samples_adjusted = counts["total_samples"] - \
            counts["mediapipe_error"]
        counts["adjusted_accuracy"] = (
            total_samples_adjusted - counts["model_error"]) / total_samples_adjusted

        # Adiciona a acurácia original ao JSON
        counts["original_accuracy"] = self.original_accuracy

        # Salva o resumo em JSON
        with open(SUMMARY_FILE, 'w') as json_file:
            json.dump(counts, json_file, indent=4)


if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()
