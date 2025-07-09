import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scripts import utils

def infer_graph(model, data, threshold=False,):

    '''Realiza la inferencia y devuelve el grafo inferido 
    desnormalizado de vuelta a sus escalas espaciales originales.
    En general, no se filtran por umbral las probabilidades
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    data = data.to(device)
    model = model.to(device)

    threshold_value = 0.5
    with torch.no_grad():
        edge_probs = model(data).squeeze()

    if threshold: #filtro por thershold
        data.y = (edge_probs > threshold_value).long()     
    else:
        data.y = edge_probs
    
    data = utils.denormalize_graph(data)
    return data

def resolve_conflicts(df):
    """
    Asegura que cada hit_id pertenezca solo a un track.
    Si un hit aparece en múltiples track_id, se elige el track con más hits y
    se descarta el resto
    """

    track_counts = df['track_id'].value_counts().to_dict() #numeor de hits por track
    
    duplicated_hits = df[df.duplicated('hit_id', keep=False)]['hit_id'].unique()
    
    tracks_to_remove = set()

    for hit in duplicated_hits:
        involved_tracks = df[df['hit_id'] == hit]['track_id'].unique()
        sorted_tracks = sorted(involved_tracks, key=lambda t: track_counts[t], reverse=True)
        #el primero es el de mayor tamaño
        tracks_to_remove.update(sorted_tracks[1:])
    
    cleaned_df = df[~df['track_id'].isin(tracks_to_remove)].copy()
    
    return cleaned_df


def clean_abrupt_angles(df, pct):

    '''Elimina tracks con curvaturas superiores al
    valor del pct % de la curvatura media del track'''

    drop = []
    to_cart = lambda r,th,z: np.array([r*np.cos(th), r*np.sin(th), z])
    for tid, g in df.groupby('track_id', sort=False):
        if len(g)<3: continue
        pts = np.vstack(g.apply(lambda row: to_cart(row.r, row.theta, row.z), axis=1))
        vecs = np.diff(pts, axis=0)
        norms = np.linalg.norm(vecs, axis=1)
        if (norms==0).any():
            drop += list(g.index[norms==0]+1)
            continue
        angs = np.arccos(np.clip((vecs[:-1]*vecs[1:]).sum(1)/(norms[:-1]*norms[1:]), -1,1))
        bad = np.where(np.abs(angs-angs.mean())>angs.mean()*pct)[0]+1
        drop += list(g.index[bad])
    return df.drop(index=drop).reset_index(drop=True)


def get_tracks(g):
    # algoritmo CTD
    mask = g.y >= 0
    ei = g.edge_index[:, mask].cpu().numpy()   # (2, n_aristas)
    probs = g.y[mask].cpu().numpy()            # (n_aristas,)

    nodes = pd.DataFrame({
        'idx':    np.arange(g.x.size(0)),
        'hit_id': g.hit_id.cpu().numpy(),
        'r':      g.x[:,0].cpu().numpy(),
        'theta':  g.x[:,1].cpu().numpy(),
        'z':      g.x[:,2].cpu().numpy(),
        'layer':  g.layer.cpu().numpy()
    })

    #mapa [índices en ei: probabilidades]
    adj = defaultdict(list)
    for edge_idx, u in enumerate(ei[0]):
        adj[int(u)].append(edge_idx)

    def get_candidate(_, layer, tracks, t): # toma segmento con mayor probabilidad
        last = tracks[t][-1]
        idxs = np.array(adj[last], dtype=int)  #índices de aristas salientes
        if idxs.size == 0:
            return None
        
        p = probs[idxs]               
        dests = ei[1, idxs]                    

        best = idxs[np.argmax(p)] # el de mayor prob
        return int(ei[1, best])

    #incializacion de tracks en 3 primeras capas
    origin = nodes[nodes['layer'].isin([0,1,2])]['idx'].tolist()
    tracks = [[v] for v in origin]

    for layer in range(10):
        for t in range(len(tracks)):
            sel = get_candidate(None, layer, tracks, t)
            if sel is not None:
                tracks[t].append(sel)

    #filtro nhits>=5
    tracks = [trk for trk in tracks if len(trk) >= 5]
    if not tracks:
        return None

    df = pd.DataFrame([
        {'idx': node, 'track_id': ti}
        for ti, trk in enumerate(tracks)
        for node in trk
    ])
    df = df.merge(
        nodes[['idx','hit_id','r','theta','z']],
        on='idx', how='left'
    )

    # limpiamos y resolvemos hits compartidos
    df = clean_abrupt_angles(df[['hit_id','r','theta','z','track_id']], 0.8)
    return resolve_conflicts(df)







