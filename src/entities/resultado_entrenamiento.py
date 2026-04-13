
class ResultadoEntrenamiento:
    def __init__(self):
        # Registro del error y precision por epoca
        self.registro_metricas = None

        # Pesos al inicio y al final del entrenamiento
        self.pesos_iniciales = []
        self.pesos_finales = []

        # Lista con los pesos capturados en cada epoca
        self.historial_pesos = []

    def guardar_metricas(self, resultado_fit):
        """Almacena las metricas devueltas por model.fit() de Keras."""
        self.registro_metricas = resultado_fit.history

    def capturar_pesos(self, pesos_actuales):
        """Registra una copia de los pesos actuales en el historial."""
        copia = [capa_w.copy() for capa_w in pesos_actuales]
        self.historial_pesos.append(copia)
