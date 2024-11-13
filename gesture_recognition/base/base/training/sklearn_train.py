from sklearn.metrics import accuracy_score
import numpy as np
import joblib


def train_model(model, dataset_train, dataset_val, model_path):
    """
    Treina um modelo sklearn usando datasets fornecidos.
    """
    # Preparar os dados do dataset
    X_train, y_train = [], []
    for data in dataset_train:
        landmarks, labels = data[0][0].numpy(), data[1].numpy()
        X_train.append(landmarks.flatten())
        y_train.append(labels)

    X_val, y_val = [], []
    for data in dataset_val:
        landmarks, labels = data[0][0].numpy(), data[1].numpy()
        X_val.append(landmarks.flatten())
        y_val.append(labels)

    # Converter para numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validação Acurácia: {accuracy:.4f}')

    joblib.dump(model, model_path)

    return model
