# TrackReco_TFG


Este proyecto está enfocado en la reconstrucción de trayectorias de partículas en el detector CMS del LHC utilizando Redes Neuronales Gráficas (GNN). La metodología implementada aprovecha las propiedades estructurales de los datos de los hits en el tracker para construir grafos que representan posibles trayectorias, y entrena una GNN para predecir las conexiones entre hits y reconstruir así las trayectorias completas de las partículas. 

---

## Estructura del repositorio

El repositorio contiene archivos **Jupyter notebook (.ipynb)** que orquestan y combinan los scripts (**.py**) encargados de albergar la implementación. Se describe a continuación la jerarquía del proyecto y su contenido:

- **TrackReco_TFG/**
  - **data/**
    - **events/**: Contiene los eventos directamente descargados de TrackML Challenge [^1].
    - **dataset_graphs_real/**: Set de grafos para entrenamiento y validación generados con un filtro de momentos realista.
    - **dataset_graphs_ideal/**: Set de grafos para entrenamiento y validación generados con un filtro de momentos ideal.
  - **configs/**: Guarda un archivo `.yaml` para configurar los hiperparámetros del procesamiento de datos y del entrenamiento del modelo.
  - **models/**: Contiene los dos modelos entrenados y los gráficos de métricas.
  - **scripts/**: Alberga toda la implementación de las tres partes del algoritmo.
    - `data_prep.py`: Procesamiento de datos y construcción de grafos.
    - `gnn_model.py`: Definición del modelo GNN.
    - `reconstruct.py`: Algoritmo CTD.
    - `utils.py`: Herramientas recurrentes: normalización y desnormalización, concatenado de grafos, carga de hiperparámetros, etc.
    - `visualize.py`: Funciones de visualización de gráficos 3D y estadísticas.
  - `dataset_generator.ipynb`: Para creación del contenido de **data/**.
  - `training.ipynb`: Entrenamiento de modelos.
  - `inference.ipynb`: Inferencia y reconstrucción de trayectorias.
  - `geometry.ipynb`: Creación de las figuras mostradas en la memoria.

---

### Nota

Se han incluido unos pocos archivos en **data/** para poder ejecutar el algoritmo sin tener que descargar todo el dataset de TrackML [^1].

---

[^1]: [TrackML Challenge](https://www.kaggle.com/c/trackml-particle-identification)
