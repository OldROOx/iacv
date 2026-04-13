import os
import pandas as pd
import numpy as np
from src.modelo_mlp import PerceptronMulticapa
from src.services.generador_graficas import generar_grafica_error, generar_grafica_pesos, generar_grafica_validacion_cruzada
from src.services.validador_cruzado import ejecutar_validacion_cruzada

# Creamos la estructura de directorios para guardar resultados
os.makedirs(os.path.join('results', 'train'), exist_ok=True)
os.makedirs(os.path.join('results', 'validated'), exist_ok=True)


def leer_dataset_xor(ruta_archivo):
    """Lee el archivo CSV del XOR y separa entradas de salidas."""
    datos_crudos = pd.read_csv(ruta_archivo, header=None)
    caracteristicas = datos_crudos.iloc[:, :-1].values
    etiquetas = datos_crudos.iloc[:, -1].values
    return caracteristicas, etiquetas


def exportar_reporte_pesos(resultado_entrenamiento, ruta_csv):
    """Genera un CSV comparando los pesos iniciales vs finales de la red."""
    iniciales_planos = np.concatenate([w.flatten() for w in resultado_entrenamiento.pesos_iniciales])
    finales_planos = np.concatenate([w.flatten() for w in resultado_entrenamiento.pesos_finales])

    df_reporte = pd.DataFrame({
        'Parametro_ID': [f'Param_{i}' for i in range(len(iniciales_planos))],
        'Peso_Inicial': iniciales_planos,
        'Peso_Final': finales_planos
    })

    df_reporte.to_csv(ruta_csv, index=False)
    print(f"Reporte de pesos guardado en: {ruta_csv}")


def exportar_reporte_validacion(resumen_vc, ruta_csv):
    """Genera un CSV con los resultados de la validacion cruzada."""
    num_folds = len(resumen_vc['precision_por_fold'])

    df_reporte = pd.DataFrame({
        'Fold': [f'Fold_{i + 1}' for i in range(num_folds)] + ['Promedio', 'Desv. Estandar'],
        'Accuracy': resumen_vc['precision_por_fold'] + [resumen_vc['precision_promedio'], resumen_vc['desviacion_precision']],
        'Loss': resumen_vc['error_por_fold'] + [resumen_vc['error_promedio'], np.std(resumen_vc['error_por_fold'])]
    })

    df_reporte.to_csv(ruta_csv, index=False)
    print(f"Reporte de validacion cruzada guardado en: {ruta_csv}")


if __name__ == "__main__":
    ruta_dataset = os.path.join('datasets', 'xor.csv')

    print("1. Cargando datos...")
    if not os.path.exists(ruta_dataset):
        print(f"Error: No se encontro el dataset en {ruta_dataset}")
        exit(1)

    caracteristicas, etiquetas = leer_dataset_xor(ruta_dataset)

    # ============================================================
    # FASE 1: Entrenamiento principal del modelo
    # ============================================================
    print("\n2. Inicializando el Perceptron Multicapa...")
    mlp = PerceptronMulticapa(dimension_entrada=2, neuronas_ocultas=4)

    print("3. Entrenando el modelo...")
    resultado = mlp.entrenar(caracteristicas, etiquetas, num_epocas=150, mostrar_logs=0)

    error_final = resultado.registro_metricas['loss'][-1]
    print(f"   Entrenamiento finalizado. Loss final: {error_final:.4f}")

    print("4. Guardando artefactos del entrenamiento...")
    ruta_modelo = os.path.join('results', 'train', 'xor_model.keras')
    mlp.guardar_modelo(ruta_modelo)
    print(f"   Modelo guardado en: {ruta_modelo}")

    ruta_csv_pesos = os.path.join('results', 'pesos_reporte.csv')
    exportar_reporte_pesos(resultado, ruta_csv_pesos)

    print("5. Generando graficas de rendimiento...")
    generar_grafica_error(resultado.registro_metricas)
    generar_grafica_pesos(resultado.historial_pesos)

    # ============================================================
    # FASE 2: Validacion Cruzada (K-Fold)
    # ============================================================
    print("\n6. Ejecutando Validacion Cruzada K-Fold...")
    precisiones, errores, resumen = ejecutar_validacion_cruzada(
        caracteristicas=caracteristicas,
        etiquetas=etiquetas,
        num_folds=4,
        neuronas_ocultas=4,
        num_epocas=150
    )

    # Guardar reporte CSV de la validacion cruzada
    ruta_csv_vc = os.path.join('results', 'validated', 'reporte_validacion_cruzada.csv')
    exportar_reporte_validacion(resumen, ruta_csv_vc)

    # Generar grafica de barras de la validacion cruzada
    generar_grafica_validacion_cruzada(precisiones)

    print("\nProceso completo. Todos los artefactos estan en /results/")
