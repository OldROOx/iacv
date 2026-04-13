import tensorflow as tf
from src.entities.resultado_entrenamiento import ResultadoEntrenamiento


class RastreadorPesosEpoca(tf.keras.callbacks.Callback):
    """Callback personalizado que registra los pesos de la red al final de cada epoca."""

    def __init__(self, entidad_resultado: ResultadoEntrenamiento):
        super().__init__()
        self.referencia_resultado = entidad_resultado

    def on_train_begin(self, logs=None):
        # Guardamos los pesos aleatorios antes de cualquier ajuste
        self.referencia_resultado.pesos_iniciales = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        # Registramos los pesos al cerrar cada epoca
        self.referencia_resultado.capturar_pesos(self.model.get_weights())

    def on_train_end(self, logs=None):
        # Guardamos los pesos una vez terminado todo el entrenamiento
        self.referencia_resultado.pesos_finales = self.model.get_weights()
