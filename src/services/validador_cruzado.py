import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models
import tensorflow as tf


def construir_modelo(dimension_entrada, neuronas_ocultas, funcion_activacion='sigmoid'):
    """Construye y compila un modelo MLP desde cero (pesos aleatorios nuevos)."""
    red = models.Sequential([
        layers.Dense(neuronas_ocultas, input_dim=dimension_entrada,
                     activation=funcion_activacion, name="capa_oculta"),
        layers.Dense(1, activation='sigmoid', name="capa_salida")
    ])

    optimizador = tf.keras.optimizers.Adam(learning_rate=0.1)

    red.compile(
        optimizer=optimizador,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return red


def ejecutar_validacion_cruzada(caracteristicas, etiquetas, num_folds=4,
                                 neuronas_ocultas=4, num_epocas=150):
    """
    Ejecuta validacion cruzada K-Fold sobre el dataset.

    Para cada fold:
      1. Separa los datos en conjunto de entrenamiento y conjunto de prueba
      2. Crea un modelo nuevo (pesos aleatorios frescos)
      3. Entrena con los datos de entrenamiento
      4. Evalua con los datos de prueba del fold

    Retorna:
      precision_por_fold: lista con el accuracy de cada fold
      error_por_fold: lista con el loss de cada fold
      resumen_vc: diccionario con el resumen estadistico
    """
    divisor_kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    precision_por_fold = []
    error_por_fold = []

    dim_entrada = caracteristicas.shape[1]

    print(f"\n{'=' * 60}")
    print(f"  VALIDACION CRUZADA - {num_folds} Folds")
    print(f"{'=' * 60}")

    for idx_fold, (indices_entrenamiento, indices_prueba) in enumerate(divisor_kfold.split(caracteristicas)):
        print(f"\n--- Fold {idx_fold + 1}/{num_folds} ---")

        # Separar datos segun los indices generados por KFold
        x_entrenamiento = caracteristicas[indices_entrenamiento]
        y_entrenamiento = etiquetas[indices_entrenamiento]
        x_prueba = caracteristicas[indices_prueba]
        y_prueba = etiquetas[indices_prueba]

        print(f"  Datos de entrenamiento: {len(indices_entrenamiento)} muestras")
        print(f"  Datos de prueba:        {len(indices_prueba)} muestras")

        # Crear un modelo NUEVO para cada fold (pesos aleatorios frescos)
        modelo_fold = construir_modelo(dim_entrada, neuronas_ocultas)

        # Entrenar con los datos del fold actual
        modelo_fold.fit(
            x_entrenamiento, y_entrenamiento,
            epochs=num_epocas,
            verbose=0
        )

        # Evaluar con los datos de prueba del fold
        perdida_eval, precision_eval = modelo_fold.evaluate(x_prueba, y_prueba, verbose=0)

        error_por_fold.append(perdida_eval)
        precision_por_fold.append(precision_eval)

        print(f"  Resultado -> Loss: {perdida_eval:.4f} | Accuracy: {precision_eval:.4f}")

    # Resumen estadistico
    promedio_precision = np.mean(precision_por_fold)
    desviacion_precision = np.std(precision_por_fold)
    promedio_error = np.mean(error_por_fold)

    resumen_vc = {
        'precision_promedio': promedio_precision,
        'desviacion_precision': desviacion_precision,
        'error_promedio': promedio_error,
        'precision_por_fold': precision_por_fold,
        'error_por_fold': error_por_fold
    }

    print(f"\n{'=' * 60}")
    print(f"  RESULTADOS FINALES DE VALIDACION CRUZADA")
    print(f"{'=' * 60}")
    print(f"  Accuracy promedio:     {promedio_precision:.4f}")
    print(f"  Desviacion estandar:   {desviacion_precision:.4f}")
    print(f"  Loss promedio:         {promedio_error:.4f}")
    print(f"{'=' * 60}\n")

    return precision_por_fold, error_por_fold, resumen_vc
