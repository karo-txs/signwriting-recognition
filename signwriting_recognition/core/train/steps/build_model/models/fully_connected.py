from core.utils.path_functions import get_internal_asset_folder
from core.train.intefaces.tf_model import TFModel
import tensorflow as tf


class FullyConnectedModel(TFModel):

    def build_from_dict(self, params: dict, len_unique_classes: int):
        self.model_name = params.get("model")
        self.learning_rate = params.get("learning_rate")
        self.lr_decay = params.get("lr_decay")
        self.epochs = params.get("epochs")
        self.batch_size = params.get("batch_size")
        
        self.builded_model = self.build(
            fc_layers=params.get("fc_layers"),
            fc_units=params.get("fc_units"),
            dropout=params.get("dropout"),
            num_classes=len_unique_classes,
            learning_rate=params.get("learning_rate"),
        )

    def build(
        self,
        fc_layers: int = 2,
        fc_units: int = 128,
        dropout: float = 0.5,
        num_classes: int = 10,
        learning_rate: float = 0.001,
    ):
        """
        Constrói um modelo Fully Connected (FC) para classificar gestos a partir dos landmarks da mão.
        """
        # Input Layers
        node_features = tf.keras.Input(shape=(21, 3), name="hand_landmarks_input")
        handness_input = tf.keras.Input(shape=(1,), name="handness_input")
        landmarks_word_input = tf.keras.Input(
            shape=(21, 3), name="landmarks_word_input"
        )

        # Flatten os landmarks para uma camada densa
        x = tf.keras.layers.Flatten()(node_features)
        handness_flat = tf.keras.layers.Flatten()(handness_input)
        landmarks_word_flat = tf.keras.layers.Flatten()(landmarks_word_input)

        # Concatenar todas as entradas
        concatenated_inputs = tf.keras.layers.Concatenate()(
            [x, handness_flat, landmarks_word_flat]
        )

        # Fully Connected Layers
        for i in range(fc_layers):
            concatenated_inputs = tf.keras.layers.BatchNormalization(
                name=f"batch_normalization_fc_{i+1}"
            )(concatenated_inputs)
            concatenated_inputs = tf.keras.layers.Dense(
                fc_units, activation="relu", name=f"dense_fc_{i+1}"
            )(concatenated_inputs)
            concatenated_inputs = tf.keras.layers.Dropout(
                dropout, name=f"dropout_fc_{i+1}"
            )(concatenated_inputs)

        # Output Layer
        outputs = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="gesture_classification_output"
        )(concatenated_inputs)

        # Compilation
        model = tf.keras.Model(
            inputs=[node_features, handness_input, landmarks_word_input],
            outputs=outputs,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
