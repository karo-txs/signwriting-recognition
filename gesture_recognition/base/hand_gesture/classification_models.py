from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from base.base.utilities import getter
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import tensorflow as tf
import numpy as np


def build_embedder_fc(fc_layers: int, fc_units: int,
                      dropout: float, num_classes: int, learning_rate: float, unfrozen_layers: int = 12):
    # Pretrained model
    embedding_model = tf.keras.models.load_model(
        getter.get_internal_asset_folder("gesture_embedder/"))
    for layer in embedding_model.layers[:unfrozen_layers]:
        layer.trainable = False
    for layer in embedding_model.layers[unfrozen_layers:]:
        layer.trainable = True

    # Input Layer
    inputs = embedding_model.input

    # Hidden Layers
    x = embedding_model.output
    for i in range(fc_layers):
        index = i + 12
        x = tf.keras.layers.BatchNormalization(
            name=f'batch_normalization_{index}')(x)
        x = tf.keras.layers.Dense(
            fc_units, activation='relu', name=f'dense_{index}')(x)
        x = tf.keras.layers.Dropout(dropout, name=f'dropout_{index}')(x)

    x = tf.keras.layers.BatchNormalization(
        name=f'batch_normalization_{(index + 1)}')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout, name=f'dropout_{(index + 1)}')(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='custom_gesture_recognizer_out')(x)

    # Compilation
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_fc(fc_layers: int = 2, fc_units: int = 128,
             dropout: float = 0.5, num_classes: int = 10, learning_rate: float = 0.001):
    """
    Constrói um modelo Fully Connected (FC) para classificar gestos a partir dos landmarks da mão.
    """
    # Input Layers
    node_features = tf.keras.Input(shape=(21, 3), name='hand_landmarks_input')
    handness_input = tf.keras.Input(shape=(1,), name='handness_input')
    landmarks_word_input = tf.keras.Input(
        shape=(21, 3), name='landmarks_word_input')

    # Flatten os landmarks para uma camada densa
    x = tf.keras.layers.Flatten()(node_features)
    handness_flat = tf.keras.layers.Flatten()(handness_input)
    landmarks_word_flat = tf.keras.layers.Flatten()(landmarks_word_input)

    # Concatenar todas as entradas
    concatenated_inputs = tf.keras.layers.Concatenate()(
        [x, handness_flat, landmarks_word_flat])

    # Fully Connected Layers
    for i in range(fc_layers):
        concatenated_inputs = tf.keras.layers.BatchNormalization(
            name=f'batch_normalization_fc_{i+1}')(concatenated_inputs)
        concatenated_inputs = tf.keras.layers.Dense(
            fc_units, activation='relu', name=f'dense_fc_{i+1}')(concatenated_inputs)
        concatenated_inputs = tf.keras.layers.Dropout(
            dropout, name=f'dropout_fc_{i+1}')(concatenated_inputs)

    # Output Layer
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='gesture_classification_output')(concatenated_inputs)

    # Compilation
    model = tf.keras.Model(
        inputs=[node_features, handness_input, landmarks_word_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_fc_2(fc_layers: int = 2, fc_units: int = 128,
               dropout: float = 0.5, num_classes: int = 10, learning_rate: float = 0.001):
    """
    Constrói um modelo Fully Connected (FC) para classificar gestos a partir dos landmarks da mão.
    """
    # Input Layers
    node_features = tf.keras.Input(shape=(21, 3), name='hand_landmarks_input')
    handness_input = tf.keras.Input(shape=(1,), name='handness_input')
    landmarks_word_input = tf.keras.Input(
        shape=(21, 3), name='landmarks_word_input')

    # Flatten os landmarks para uma camada densa
    x = tf.keras.layers.Flatten()(node_features)

    # Fully Connected Layers
    for i in range(fc_layers):
        x = tf.keras.layers.BatchNormalization(
            name=f'batch_normalization_fc_{i+1}')(x)
        x = tf.keras.layers.Dense(
            fc_units, activation='relu', name=f'dense_fc_{i+1}')(x)
        x = tf.keras.layers.Dropout(
            dropout, name=f'dropout_fc_{i+1}')(x)

    # Output Layer
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='gesture_classification_output')(x)

    # Compilation
    model = tf.keras.Model(
        inputs=[node_features, handness_input, landmarks_word_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_conv_fc(fc_layers: int = 2, fc_units: int = 128,
                  dropout: float = 0.5, num_classes: int = 10, learning_rate: float = 0.001):
    """
    Constrói um modelo Fully Connected (FC) para classificar gestos a partir dos landmarks da mão.
    """
    # Input Layers
    node_features = tf.keras.Input(shape=(21, 3), name='hand_landmarks_input')
    handness_input = tf.keras.Input(shape=(1,), name='handness_input')
    landmarks_word_input = tf.keras.Input(
        shape=(21, 3), name='landmarks_word_input')

    # Flatten os landmarks para uma camada densa
    x = tf.keras.layers.Conv1D(
        64, kernel_size=3, activation='relu')(node_features)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)

    # Fully Connected Layers
    for i in range(fc_layers):
        x = tf.keras.layers.BatchNormalization(
            name=f'batch_normalization_fc_{i+1}')(x)
        x = tf.keras.layers.Dense(
            fc_units, activation='relu', name=f'dense_fc_{i+1}')(x)
        x = tf.keras.layers.Dropout(
            dropout, name=f'dropout_fc_{i+1}')(x)

    # Output Layer
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='gesture_classification_output')(x)

    # Compilation
    model = tf.keras.Model(
        inputs=[node_features, handness_input, landmarks_word_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_conv_embedder(fc_layers: int, fc_units: int,
                        dropout: float, num_classes: int,
                        learning_rate: float, unfrozen_layers: int = 12,
                        conv_layers: int = 2, conv_filters: int = 64):
    # Pretrained model
    embedding_model = tf.keras.models.load_model(
        getter.get_internal_asset_folder("gesture_embedder/"))
    for layer in embedding_model.layers[:unfrozen_layers]:
        layer.trainable = False
    for layer in embedding_model.layers[unfrozen_layers:]:
        layer.trainable = True

    # Input Layer
    inputs = embedding_model.input

    # Reshape output to add the extra dimension required by Conv1D
    x = tf.keras.layers.Reshape(
        (embedding_model.output_shape[1], 1))(embedding_model.output)

    # Convolutional Layers
    for i in range(conv_layers):
        x = tf.keras.layers.Conv1D(
            filters=conv_filters, kernel_size=3, activation='relu', name=f'conv_{i+1}')(x)
        x = tf.keras.layers.BatchNormalization(
            name=f'batch_norm_conv_{i+1}')(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=2, name=f'max_pool_{i+1}')(x)
        conv_filters *= 2  # Dobrar o número de filtros após cada camada

    x = tf.keras.layers.Flatten()(x)

    # Fully Connected Layers
    for i in range(fc_layers):
        index = i + 12 + conv_layers
        x = tf.keras.layers.BatchNormalization(
            name=f'batch_normalization_{index}')(x)
        x = tf.keras.layers.Dense(
            fc_units, activation='relu', name=f'dense_{index}')(x)
        x = tf.keras.layers.Dropout(dropout, name=f'dropout_{index}')(x)

    x = tf.keras.layers.BatchNormalization(
        name=f'batch_normalization_{(index + 1)}')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout, name=f'dropout_{(index + 1)}')(x)

    # Output layer
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='custom_gesture_recognizer_out')(x)

    # Compilation
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_conv(conv_layers: int = 2, conv_filters: int = 64,
               fc_layers: int = 2, fc_units: int = 128,
               dropout: float = 0.5, num_classes: int = 10,
               learning_rate: float = 0.001):
    # Input Layer (Tupla de inputs: landmarks, handness, landmarks_word)
    inputs = tf.keras.Input(shape=(21, 3), name='hand_landmarks_input')
    handness_input = tf.keras.Input(shape=(1,), name='handness_input')
    landmarks_word_input = tf.keras.Input(
        shape=(1,), name='landmarks_word_input')

    # Utiliza apenas os landmarks como entrada para a rede convolucional
    x = inputs

    # Convolutional Layers
    for i in range(conv_layers):
        x = tf.keras.layers.Conv1D(
            filters=conv_filters, kernel_size=min(3, x.shape[1]), activation='relu', name=f'conv_{i+1}')(x)
        x = tf.keras.layers.BatchNormalization(
            name=f'batch_norm_conv_{i+1}')(x)
        if x.shape[1] > 1:
            x = tf.keras.layers.MaxPooling1D(
                pool_size=2, name=f'max_pool_{i+1}')(x)
        conv_filters *= 2  # Dobrar o número de filtros após cada camada

    # Aplicar Global Average Pooling para evitar redução excessiva
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layers
    for i in range(fc_layers):
        x = tf.keras.layers.BatchNormalization(
            name=f'batch_normalization_fc_{i+1}')(x)
        x = tf.keras.layers.Dense(
            fc_units, activation='relu', name=f'dense_fc_{i+1}')(x)
        x = tf.keras.layers.Dropout(dropout, name=f'dropout_fc_{i+1}')(x)

    # Output Layer
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='gesture_classification_output')(x)

    # Compilation
    model = tf.keras.Model(
        inputs=[inputs, handness_input, landmarks_word_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_hand_adjacency_matrix():
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
        (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
        (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
        (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
        (0, 17), (17, 18), (18, 19), (19, 20)  # Mínimo
    ]
    adjacency_matrix = np.zeros((21, 21), dtype=np.float32)

    for start, end in connections:
        adjacency_matrix[start, end] = 1.0
        adjacency_matrix[end, start] = 1.0
    return tf.convert_to_tensor(adjacency_matrix, dtype=tf.float32)


def build_lstm(lstm_units: int = 64, lstm_layers: int = 2, fc_layers: int = 2, fc_units: int = 128,
               dropout: float = 0.5, num_classes: int = 10, learning_rate: float = 0.001):
    """
    Constrói um modelo LSTM para classificar gestos a partir dos landmarks da mão.
    """
    # Input Layer
    node_features = tf.keras.Input(shape=(21, 3), name='hand_landmarks_input')
    handness_input = tf.keras.Input(shape=(1,), name='handness_input')
    landmarks_word_input = tf.keras.Input(
        shape=(21, 3), name='landmarks_word_input')

    # Aplicar camadas LSTM nos landmarks
    x = node_features
    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        x = tf.keras.layers.LSTM(
            units=lstm_units, return_sequences=return_sequences, name=f'lstm_{i+1}')(x)
        lstm_units *= 2  # Dobrar o número de unidades após cada camada

    # Flatten outras entradas
    handness_flat = tf.keras.layers.Flatten()(handness_input)
    landmarks_word_flat = tf.keras.layers.Flatten()(landmarks_word_input)

    # Concatenar todas as entradas
    concatenated_inputs = tf.keras.layers.Concatenate()(
        [x, handness_flat, landmarks_word_flat])

    # Fully Connected Layers
    for i in range(fc_layers):
        concatenated_inputs = tf.keras.layers.BatchNormalization(
            name=f'batch_normalization_fc_{i+1}')(concatenated_inputs)
        concatenated_inputs = tf.keras.layers.Dense(
            fc_units, activation='relu', name=f'dense_fc_{i+1}')(concatenated_inputs)
        concatenated_inputs = tf.keras.layers.Dropout(
            dropout, name=f'dropout_fc_{i+1}')(concatenated_inputs)

    # Output Layer
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', name='gesture_classification_output')(concatenated_inputs)

    # Compilation
    model = tf.keras.Model(
        inputs=[node_features, handness_input, landmarks_word_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_random_forest(n_estimators=200):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42)


def build_svm(c=1.0, kernel="rbf"):
    return make_pipeline(StandardScaler(), SVC(C=c, kernel=kernel))


def build_gbc(n_estimators=100, learning_rate=0.1, max_depth=3):
    return GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                      max_depth=max_depth, random_state=42)


def build_adaboost(n_estimators=100):
    return AdaBoostClassifier(n_estimators=n_estimators, random_state=42)


def build_knn(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def build_lr(max_iter=200):
    return LogisticRegression(max_iter=max_iter, random_state=42)
