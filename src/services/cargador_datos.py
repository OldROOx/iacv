import pandas as pd
import os


def leer_dataset_csv(ruta_archivo, encabezados=None):
    """Lee un archivo CSV y separa las columnas de entrada y salida."""
    datos_crudos = pd.read_csv(ruta_archivo, header=encabezados)

    # Todas las columnas menos la ultima son las entradas
    caracteristicas = datos_crudos.iloc[:, :-1].values
    # La columna final corresponde a la salida esperada
    etiquetas = datos_crudos.iloc[:, -1].values

    return caracteristicas, etiquetas


if __name__ == "__main__":
    ruta_csv = os.path.join('datasets', 'xor.csv')

    if os.path.exists(ruta_csv):
        caract, etiq = leer_dataset_csv(ruta_csv)
        print("Datos cargados exitosamente:")
        print("Entradas:\n", caract)
        print("Salidas:\n", etiq)
    else:
        print(f"Error: No se encontro el archivo en {ruta_csv}")
