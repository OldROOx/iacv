import os
import numpy as np
import tensorflow as tf


def ejecutar_prediccion():
    """Carga el modelo entrenado y ejecuta predicciones sobre las 4 combinaciones del XOR."""
    ruta_modelo = os.path.join('results', 'train', 'xor_model.keras')

    print("Iniciando prueba de prediccion...")

    if not os.path.exists(ruta_modelo):
        print(f"Error: No se encontro el modelo en {ruta_modelo}.")
        print("Ejecuta 'python app.py' primero para entrenar la red.")
        return

    # Cargar el modelo previamente entrenado
    print(f"Cargando modelo desde: {ruta_modelo}")
    modelo_entrenado = tf.keras.models.load_model(ruta_modelo)

    # Las 4 combinaciones posibles del XOR
    entradas_prueba = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    print("\nRealizando predicciones...\n")
    predicciones_crudas = modelo_entrenado.predict(entradas_prueba, verbose=0)

    # Mostrar resultados formateados
    print("--- Resultados de la Red Neuronal (XOR) ---")
    print(f"{'Entrada (X)':<15} | {'Salida Real':<15} | {'Prediccion Cruda':<20} | {'Prediccion Final'}")
    print("-" * 75)

    for idx in range(len(entradas_prueba)):
        entrada_actual = entradas_prueba[idx]
        salida_esperada = entrada_actual[0] ^ entrada_actual[1]

        # Valor directo de la sigmoide (entre 0 y 1)
        valor_sigmoide = predicciones_crudas[idx][0]

        # Redondeamos: >= 0.5 es 1, < 0.5 es 0
        prediccion_redondeada = int(round(valor_sigmoide))

        print(f"{str(entrada_actual):<15} | {salida_esperada:<15} | {valor_sigmoide:<20.4f} | {prediccion_redondeada}")


if __name__ == "__main__":
    ejecutar_prediccion()
