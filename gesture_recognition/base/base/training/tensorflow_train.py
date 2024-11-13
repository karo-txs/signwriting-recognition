

def train_model(model, dataset_train, dataset_val, callbacks, epochs=100):
    history = model.fit(x=dataset_train,
                        epochs=epochs,
                        validation_data=dataset_val,
                        callbacks=callbacks)

    return history, model
