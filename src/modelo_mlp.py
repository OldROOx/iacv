from src.entities.resultado_entrenamiento import ResultadoEntrenamiento
from src.services.rastreador_pesos import RastreadorPesosEpoca

from tensorflow.keras import layers, models
import tensorflow as tf


class PerceptronMulticapa:
    def __init__(self, dimension_entrada=2, neuronas_ocultas=4, config_activacion=None):
        self.dimension_entrada = dimension_entrada
        self.neuronas_ocultas = neuronas_ocultas
        # Si no se pasa configuracion, usamos sigmoid por defecto (ideal para XOR)
        self.funcion_activacion = config_activacion.get_activation() if config_activacion else 'sigmoid'
        self.red = self._construir_arquitectura()
        self.resultado = ResultadoEntrenamiento()

    def _construir_arquitectura(self):
        """Construye la topologia: Entrada -> Oculta -> Salida."""
        red = models.Sequential([
            # Capa oculta con activacion no lineal para resolver XOR
            layers.Dense(self.neuronas_ocultas,
                         input_dim=self.dimension_entrada,
                         activation=self.funcion_activacion,
                         name="capa_oculta"),
            # Capa de salida con 1 neurona para clasificacion binaria
            layers.Dense(1, activation='sigmoid', name="capa_salida")
        ])

        optimizador_rapido = tf.keras.optimizers.Adam(learning_rate=0.1)

        red.compile(
            optimizer=optimizador_rapido,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return red

    def entrenar(self, caracteristicas, etiquetas, num_epocas=500, mostrar_logs=0):
        """Ejecuta el ciclo completo de entrenamiento del modelo."""
        print(f"Iniciando entrenamiento por {num_epocas} epocas...")

        # Instanciamos el rastreador de pesos
        rastreador = RastreadorPesosEpoca(self.resultado)

        # Ejecutamos el entrenamiento de Keras
        resultado_fit = self.red.fit(
            caracteristicas, etiquetas,
            epochs=num_epocas,
            verbose=mostrar_logs,
            callbacks=[rastreador]
        )

        # Almacenamos las metricas de entrenamiento
        self.resultado.guardar_metricas(resultado_fit)
        return self.resultado

    def guardar_modelo(self, ruta_guardado):
        """Guarda el modelo entrenado en formato .keras."""
        self.red.save(ruta_guardado)
