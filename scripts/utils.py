import pandas as pd
import yaml
import os
import numpy as np


def concatenar_track_ids(lista_dfs):
    concatenados = []
    offset = 0
    for df in lista_dfs:
        df_copy = df.copy()
        df_copy['track_id'] = df_copy['track_id'] + offset
        concatenados.append(df_copy)
        offset = df_copy['track_id'].max() + 1
    return pd.concat(concatenados, ignore_index=True)

def load_hyper(name):
    """
    Carga de hiperparámetros desde archivo yaml. Maneja los valores string que involucran pi
    """
    config_path = f"configs/{name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    def _parse(v):
        if isinstance(v, str):
            expr = v.replace(" ", "")
            if expr == "pi":
                return np.pi
            if expr.startswith("pi/"):
                return np.pi / float(expr.split("/", 1)[1])
            if expr.startswith("pi*"):
                return np.pi * float(expr.split("*", 1)[1])
            try:
                return float(expr)
            except ValueError:
                raise ValueError(f"No se pudo interpretar '{v}' como número o expresión con pi.")
        return v

    modelo_cfg = config.get("modelo", {})
    procesamiento_cfg = {
        k: _parse(val) 
        for k, val in config.get("procesamiento", {}).items()}

    return modelo_cfg, procesamiento_cfg



def normalize_graph(graph):
    '''Normaliza r y z considerando sus valores máximos y mínimos.
    (el resto de atributos se mueven en escalas parecidas)'''

    node_ranges = {'r': (0, 1500), 'z': (-1120, 1120)}

    x = graph.x.clone().float()

    if x.shape[1] >= len(node_ranges):
        for i, k in enumerate(node_ranges):
            mn, mx = node_ranges[k]
            x[:, i] = (x[:, i] - mn) / (mx - mn)
        graph.x = x

    return graph


def denormalize_graph(graph):

    '''Desnormalización de r y z'''

    node_ranges = {'r': (0, 1500), 'z': (-1120, 1120)}

    x = graph.x.clone().float()
    if x.shape[1] >= len(node_ranges):
        for i, k in enumerate(node_ranges):
            mn, mx = node_ranges[k]
            x[:, i] = x[:, i] * (mx - mn) + mn
        graph.x = x  
    return graph

def align_track_id(pred_tracks, ground_tracks):

    '''Para un track reconstruido que coincide con su track solución,
     esta función da el mismo track_id a ambos'''

    gt_map = ground_tracks.set_index('hit_id')['track_id']
    aux = pred_tracks[['hit_id', 'track_id']].copy()
    aux['gt_id'] = aux['hit_id'].map(gt_map)
    mode_map = aux.dropna().groupby('track_id')['gt_id'] \
                  .agg(lambda x: x.value_counts().idxmax()).to_dict()

    pred_tracks = pred_tracks[pred_tracks['track_id'].isin(mode_map)].copy()
    pred_tracks['track_id'] = pred_tracks['track_id'].map(mode_map).astype(int)

    used = set(pred_tracks['track_id'])
    ground_tracks = ground_tracks[ground_tracks['track_id'].isin(used)].copy().reset_index(drop=True)

    return pred_tracks.reset_index(drop=True), ground_tracks