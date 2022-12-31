import tensorflow as tf
from transformers import (
    TFBertModel,
    TFRobertaModel,
)
import numpy as np 


from tensorflow.keras.layers import ( #type: ignore
    Dropout, 
    Dense,
    Input,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Sequential, Model #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore


class TensorFlowExperiment(TensorFlowExperimentUtils):
    def __init__(self, exp_name):
        super(TensorFlowExperimentUtils, self).__init__()
        self.exp_name = exp_name
        self.strategy = tf.distribute.get_strategy()
        self.earlystopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)
        self.learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                                    patience=3, 
                                                    verbose=1, 
                                                    factor=0.3, 
                                                    min_lr=0.0001)
        self.model_checkpoint = 'bert-base-uncased'

    def create_classification_model_single_text(
        self, bert_model = 'robert', loss="bce", optim="adam", learning_rate=0.0001
    ):
        with self.strategy.scope():
            transformer_model = TFBertModel.from_pretrained(self.model_checkpoint) if bert_model=='bert' else TFRobertaModel.from_pretrained(self.model_checkpoint) 
            input_ids = Input(shape=(None,), name="input_ids1", dtype="int32")
            input_mask = Input(shape=(None,), name="attention_mask1", dtype="int32")

            embedding = transformer_model(input_ids, attention_mask=input_mask).last_hidden_state
            embedding = GlobalAveragePooling1D()(embedding)

            x = Dropout(0.3)(embedding)
            x = Dense(512, activation="relu")(x)
            output = Dense(1, activation="sigmoid")(x)

            model = Model(inputs=[input_ids, input_mask], outputs=output)

            for layer in model.layers[:3]:
                layer.trainable = False

            model_loss = (
                tf.keras.losses.MeanSquaredError()
                if loss == "bce"
                else tf.keras.losses.BinaryCrossentropy()
            )
            optimizer = (
                tf.keras.optimizers.Adam(learning_rate=learning_rate)
                if optim == "adam"
                else tf.keras.optimizers.SGD(learning_rate=learning_rate)
            )
            
            model.compile(
                loss=model_loss,
                optimizer=optimizer,
                metrics=["accuracy"],
            )

        model.summary()
        return model

    def create_model_text_pair(
        self,
        bert_model='robert',
        distance_metric="l1",
        task_type="classification",
        loss="bce",
        optim="adam",
        learning_rate=0.0001,
    ):
        with self.strategy.scope():
            transformer_model = TFBertModel.from_pretrained(self.model_checkpoint) if bert_model=='bert' else TFRobertaModel.from_pretrained(self.model_checkpoint)
            input_ids1 = Input(shape=(None,), name="input_ids1", dtype="int32")
            input_ids2 = Input(shape=(None,), name="input_ids2", dtype="int32")
            input_mask1 = Input(shape=(None,), name="attention_mask1", dtype="int32")
            input_mask2 = Input(shape=(None,), name="attention_mask2", dtype="int32")

            embedding1 = transformer_model(input_ids1, attention_mask=input_mask1).last_hidden_state
            embedding2 = transformer_model(input_ids2, attention_mask=input_mask2).last_hidden_state

            dist = (
                L1Dist()(embedding1, embedding2)
                if distance_metric == "l1"
                else L2Dist()(embedding1, embedding2)
            )

            x = Dropout(0.3)(dist)
            x = Dense(512, activation="relu")(x)
            output = (
                Dense(1, activation="sigmoid")(x)
                if task_type == "classification"
                else Dense(1, activation="linear")(x)
            )

            model = Model(inputs=[input_ids1, input_mask1, input_ids2, input_mask2], outputs=output)

            for layer in model.layers[:5]:
                layer.trainable = False

            model_loss = (
                tf.keras.losses.MeanSquaredError()
                if loss == "bce"
                else tf.keras.losses.BinaryCrossentropy()
            )
            optimizer = (
                tf.keras.optimizers.Adam(learning_rate=learning_rate)
                if optim == "adam"
                else tf.keras.optimizers.SGD(learning_rate=learning_rate)
            )
            model.compile(
                loss=model_loss,
                optimizer=optimizer,
                metrics=["accuracy"],
            )
            return model
    
    def train(self, params):
        bert_model=params['bert_model']
        approach=params['approach']
        distance_metric=params['distance_metric']
        task_type=params['task_type']
        loss=params['loss']
        optim=params['optim']
        learning_rate=params['learning_rate']
        epochs=params['epochs']
        batch_size=params['batch_size']

        X1_train=params['X1_train']
        X2_train=params['X2_train']
        Y_train=params['Y_train']
        X1_test=params['X1_test']
        X2_test=params['X2_test']
        Y_test=params['Y_test'] # these can be also None 

        model = self.create_classification_model_single_text(
            bert_model=bert_model, loss=loss, optim=optim, learning_rate=learning_rate
        ) if approach=='single_text' else self.create_model_text_pair(
            bert_model=bert_model,
            distance_metric=distance_metric,
            task_type=task_type,
            loss=loss,optim=optim, learning_rate=learning_rate
        )

        if approach=='single_text':
            history = model.fit(
                (
                    np.asarray(X1_train["input_ids"]),
                    np.asarray(X1_train["attention_mask"])
                ),
                Y_train, 
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(
                    (
                        np.asarray(X1_test["input_ids"]),
                        np.asarray(X1_test["attention_mask"]),
                    ),
                    Y_test,
            ),
            callbacks=[self.earlystopping, self.learning_rate_reduction],
        )

        else:
            history = model.fit(
            (
                np.asarray(X1_train["input_ids"]),
                np.asarray(X1_train["attention_mask"]),
                np.asarray(X2_train["input_ids"]),
                np.asarray(X2_train["attention_mask"]),
            ),
            Y_train,
            batch_size=batch_size,
            epochs=5,
            validation_data=(
                (
                    np.asarray(X1_test["input_ids"]),
                    np.asarray(X1_test["attention_mask"]),
                    np.asarray(X2_test["input_ids"]),
                    np.asarray(X2_test["attention_mask"]),
                ),
                Y_test,
            ),
            callbacks=[self.earlystopping, self.learning_rate_reduction],
        )

        return history