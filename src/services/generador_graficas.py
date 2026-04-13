import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generar_grafica_error(registro_metricas, ruta_salida='results/grafica_error.png'):
    """
    Genera la grafica de evolucion del error (loss).
    Si existe validation loss, la grafica tambien para detectar sobreentrenamiento.
    """
    total_epocas = range(1, len(registro_metricas['loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(total_epocas, registro_metricas['loss'], 'b-', label='Error de Entrenamiento (Loss)')

    # Incluimos la curva de validacion si fue generada durante el entrenamiento
    if 'val_loss' in registro_metricas:
        plt.plot(total_epocas, registro_metricas['val_loss'], 'r--', label='Error de Validacion (Val Loss)')

    plt.title('Evolucion del Error y Deteccion de Sobreentrenamiento')
    plt.xlabel('Epocas')
    plt.ylabel('Loss (Binary Crossentropy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"Grafica de error guardada en: {ruta_salida}")


def generar_grafica_pesos(historial_pesos, ruta_salida='results/grafica_pesos.png'):
    """
    Genera una grafica mostrando como cada peso individual cambia a traves de las epocas.
    """
    num_epocas = len(historial_pesos)
    total_parametros = len(np.concatenate([w.flatten() for w in historial_pesos[0]]))

    # Matriz donde cada fila es una epoca y cada columna un parametro
    matriz_parametros = np.zeros((num_epocas, total_parametros))

    for ep in range(num_epocas):
        aplanados = np.concatenate([w.flatten() for w in historial_pesos[ep]])
        matriz_parametros[ep, :] = aplanados

    plt.figure(figsize=(12, 6))

    for idx_param in range(total_parametros):
        plt.plot(range(1, num_epocas + 1), matriz_parametros[:, idx_param], alpha=0.6)

    plt.title('Evolucion de los Pesos de la Red a lo largo del Entrenamiento')
    plt.xlabel('Epocas')
    plt.ylabel('Valor del Peso')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"Grafica de pesos guardada en: {ruta_salida}")


def generar_grafica_validacion_cruzada(puntajes_folds, ruta_salida='results/validated/grafica_validacion_cruzada.png'):
    """
    Genera una grafica de barras con el accuracy obtenido en cada fold
    de la validacion cruzada, junto con la linea del promedio.
    """
    num_folds = len(puntajes_folds)
    etiquetas_folds = [f'Fold {i + 1}' for i in range(num_folds)]
    promedio = np.mean(puntajes_folds)

    plt.figure(figsize=(8, 5))
    barras = plt.bar(etiquetas_folds, puntajes_folds, color='#2E86C1', edgecolor='#1A5276', width=0.5)

    # Linea horizontal del promedio
    plt.axhline(y=promedio, color='#E74C3C', linestyle='--', linewidth=2,
                label=f'Promedio: {promedio:.4f}')

    # Etiquetas de valor sobre cada barra
    for barra, puntaje in zip(barras, puntajes_folds):
        plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.005,
                 f'{puntaje:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('Accuracy por Fold - Validacion Cruzada (K-Fold)')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    print(f"Grafica de validacion cruzada guardada en: {ruta_salida}")
