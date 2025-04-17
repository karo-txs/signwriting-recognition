from core.utils.path_functions import get_internal_asset_folder
from core.train.intefaces.tf_model import TFModel
import tensorflow as tf


class FullyConnectedEmbedderModel(TFModel):

    def build_from_dict(self, params: dict, len_unique_classes: int):
        self.model_name = params.get("model")
        self.learning_rate = params.get("learning_rate")
        self.lr_decay = params.get("lr_decay")
        self.epochs = params.get("epochs")
        self.batch_size = params.get("batch_size")
        
        self.builded_model = self.build(
            unfrozen_layers=params.get("unfrozen_layers"),
            fc_layers=params.get("fc_layers"),
            fc_units=params.get("fc_units"),
            dropout=params.get("dropout"),
            num_classes=len_unique_classes,
            learning_rate=params.get("learning_rate"),
        )

    def build(
        self,
        fc_layers: int,
        fc_units: int,
        dropout: float,
        num_classes: int,
        learning_rate: float,
        unfrozen_layers: int = 12,
    ):
        # Pretrained model
        embedding_model = tf.keras.models.load_model(
            get_internal_asset_folder("gesture_embedder/")
        )
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
            x = tf.keras.layers.BatchNormalization(name=f"batch_normalization_{index}")(
                x
            )
            x = tf.keras.layers.Dense(
                fc_units, activation="relu", name=f"dense_{index}"
            )(x)
            x = tf.keras.layers.Dropout(dropout, name=f"dropout_{index}")(x)

        x = tf.keras.layers.BatchNormalization(
            name=f"batch_normalization_{(index + 1)}"
        )(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout, name=f"dropout_{(index + 1)}")(x)

        # Output layer
        outputs = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="custom_gesture_recognizer_out"
        )(x)

        # Compilation
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
