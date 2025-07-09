import pandas as pd
import numpy as np
import torch as torch
from torch_geometric.data import Data
from scripts import utils


azimutal_cut = 0.35

def angular_space_divisor(dtheta, dphi):
    """
    Divide el espacio en regiones esféricas y devuelve 
    una lista con los ángulos theta y phi que las limitan
    """
    if not (0 < dtheta <= 2*np.pi) or not (0 < dphi <= np.pi):
        raise ValueError("0 < dtheta ≤ 2π y 0 < dphi ≤ π")

    thetas = np.arange(-np.pi, np.pi, dtheta) # theta entre +-pi
    if thetas[-1] < np.pi:
        thetas = np.append(thetas, np.pi)
    phis = np.arange(0, np.pi, dphi) # phi entre 0 y pi
    if phis[-1] < np.pi:
        phis = np.append(phis, np.pi)

    return [
        (t0, t1, p0, p1)
        for t0, t1 in zip(thetas[:-1], thetas[1:])
        for p0, p1 in zip(phis[:-1],   phis[1:])]


def space_batches(hits, truth, particles, n_event, dtheta, dphi):

    '''Toma los hits de las capas cilindricas del dataset y los empaqueta en
    regiones espaciales. Si dtheta = 2pi y dphi = pi, no se produce división espacial'''

    particles['pt'] = np.sqrt(particles.px**2 + particles.py**2)
    truth = (truth[['hit_id', 'particle_id']].merge(particles[['particle_id', 'pt']], on='particle_id')) # unimos los dfs

    barrels = [(8,2), (8,4), (8,6), (8,8), (13,2), (13,4), (13,6), (13,8), (17,2), (17,4)] 
    groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([groups.get_group(barrels[i]).assign(layer=i) for i in range(len(barrels))]) # identificadores ordinales de capas cilíndricas
    hits['r'], hits['theta'] = np.sqrt(hits.x**2 + hits.y**2), np.arctan2(hits.y, hits.x)
    hits['phi'] = np.arccos(hits.z/np.sqrt(hits.r**2+hits.z**2))
    hits = hits[(hits['phi'] >= azimutal_cut) & (hits['phi'] <= np.pi - azimutal_cut)] #tomo particulas que pasan por 10 capas

    hits = (hits[['hit_id', 'r', 'theta', 'z', 'phi', 'layer']].merge(truth[['hit_id', 'particle_id', 'pt']], on='hit_id')) #añadimos pt
    hits = hits[['hit_id', 'particle_id', 'r', 'theta', 'z', 'phi', 'pt', 'layer']] 
    hits = hits.assign(n_event=int(n_event))
    
    cells = angular_space_divisor(dtheta, dphi) # lista con las celdas dtheta, dphi espaciales
    cones = []
    selection_cones_axis_angles = [] # guarda el angulo del eje de los conos de selección para la conexion de nodos en segments

    for theta_min, theta_max, phi_min, phi_max in cells:
        mask_theta = (hits['theta'] >= theta_min) & (hits['theta'] < theta_max)

        mask_phi   = (hits['phi']   >= phi_min)   & (hits['phi']   < phi_max)
        sub_hits = hits[mask_phi & mask_theta]
        
        if not sub_hits.empty:
            cones.append(sub_hits)
            selection_cones_axis_angles.append(abs(phi_max+phi_min)/2)   
    return cones


def get_segments_batches(hits, pt_min=0.5, pt_max=1000):

    '''Crea y filtra los segmentos que formarán el grafo. 
    Si pt_min = 0, únicamente actúa el filtro de momentos implementado.
    Si pt_min > 0.5, se aplica el filtro ideal que no representa la situaciones
    de reconstrucción que ocurren en la realidad'''

    hits = hits.reset_index(drop=True).copy()
    hits['orig_idx'] = np.arange(len(hits))

    dap = 0.141371669
    cone_angles = { #aperturas por capa de los conos de selección
        0: 0.7109920216019006 - 2*dap,
        1: 0.3141592653589793,
        2: 0.3141592653589793,
        3: 0.3141592653589793,
        4: 0.38029805806613287,
        5: 0.5787144361875936,
        6: 0.7109920216019006,
        7: 0.9094083997233612,
        8: 1.1078247778448218,
        9: 0
    }

    N = len(hits)
    alive_mask = np.zeros(N, dtype=bool)
    alive_mask[hits['layer'].to_numpy() == 0] = True #hits dentro del cono de seleccion de la capa anterior, no descartados

    connections = []

    for inner_layer in range(0, 10):
        outer_layer = inner_layer + 1

        layer_arr = hits['layer'].to_numpy() # para hits no descartados en capa interna
        idx_inner = np.nonzero(alive_mask & (layer_arr == inner_layer))[0]
        if idx_inner.size == 0:
            break
        idx_outer = np.nonzero(layer_arr == outer_layer)[0]

        inner = hits.iloc[idx_inner].reset_index(drop=True)
        outer = hits.iloc[idx_outer].reset_index(drop=True)

        pairs = pd.merge(inner, outer, on='n_event', suffixes=('_in','_out')) #todos los segmentos entre las dos capas

        pairs['phi_in']  = ((pairs.theta_in  + np.pi) % (2*np.pi)) - np.pi
        pairs['phi_out'] = ((pairs.theta_out + np.pi) % (2*np.pi)) - np.pi
        eta_in  = -np.log(np.tan(0.5 * np.arccos(
            pairs.z_in  / np.sqrt(pairs.r_in**2  + pairs.z_in**2 )
        )))
        eta_out = -np.log(np.tan(0.5 * np.arccos(
            pairs.z_out / np.sqrt(pairs.r_out**2 + pairs.z_out**2)
        )))
        #atributos de las aristas que toma el modelo
        pairs['delta_eta'] = eta_out - eta_in
        pairs['delta_phi'] = ((pairs.phi_out - pairs.phi_in + np.pi) % (2*np.pi)) - np.pi #cambio de nombres, phi se refiere a theta

        # cono de selección, apertura y orientación
        r1, r2 = pairs.r_in.to_numpy(), pairs.r_out.to_numpy()
        z1, z2 = pairs.z_in.to_numpy(), pairs.z_out.to_numpy()
        x1 = r1 * np.cos(pairs.theta_in)
        y1 = r1 * np.sin(pairs.theta_in)
        x2 = r2 * np.cos(pairs.theta_out)
        y2 = r2 * np.sin(pairs.theta_out)

        dx, dy, dz = x2-x1, y2-y1, z2-z1
        dot   = x1*dx + y1*dy + z1*dz
        norm1 = x1**2 + y1**2 + z1**2
        norm2 = dx**2 + dy**2 + dz**2
        cos_theta = dot / (np.sqrt(norm1*norm2) + 1e-9)

        #la apertura disminuye al ser paralelo a z
        cos_z = z1 / (np.sqrt(norm1) + 1e-9)
        sin_z = np.sqrt(np.clip(1 - cos_z**2, 0, 1))
        eff_ap  = cone_angles[inner_layer] * sin_z
        cos_alpha = np.cos(eff_ap/2)

        mask_cone = cos_theta >= cos_alpha
        filtered = pairs.loc[mask_cone].copy()

        filtered['label'] = (
            (filtered.particle_id_in == filtered.particle_id_out) &
            (filtered.particle_id_in != 0))
        
        if pt_min == 0:
            def calc_dtheta(theta1, theta2):
                dphi = theta2 - theta1
                dphi[dphi > np.pi] -= 2*np.pi
                dphi[dphi < -np.pi] += 2*np.pi
                return dphi
            #filtro de momentos
            dtheta = calc_dtheta(pairs.phi_in, pairs.phi_out)
            dz = pairs.z_out - pairs.z_in
            dr = pairs.r_out - pairs.r_in
            var_theta = dtheta / dr
            z0 = pairs.z_in - pairs.r_in * dz / dr

            filtered = filtered.assign(var_theta=var_theta[mask_cone], z0=z0[mask_cone])

            def filter_segments(segments, theta_slope_min = 0., theta_slope_max = 0.0006, z0_max = 100):
                sel_mask = ((segments.var_theta.abs() > theta_slope_min) &
                            (segments.var_theta.abs() < theta_slope_max) &
                            (segments.z0.abs() < z0_max))
                return segments.assign(selected=sel_mask)
            
            filtered = filter_segments(filtered)
        
        connections.append(filtered[[
            'n_event',
            'hit_id_in','particle_id_in','layer_in','r_in','theta_in','z_in','pt_in',
            'hit_id_out','particle_id_out','layer_out','r_out','theta_out','z_out','pt_out',
            'delta_eta','delta_phi','label'
        ]])


        #actualizar la selección de hits emisores en la siguiente iter
        alive_mask[ filtered['orig_idx_out'].to_numpy() ] = True

    #filtro ideal de momentos (no tiene efecto si pt_min = 0, filtro real)
    segments = pd.concat(connections, ignore_index=True)
    mask_pt = (
        (segments.pt_in  >= pt_min) & (segments.pt_in  <= pt_max) &
        (segments.pt_out >= pt_min) & (segments.pt_out <= pt_max)
    )
    return segments.loc[mask_pt].reset_index(drop=True)



def graph_builder(segments, drop_y=False):

    '''A partir de los segmentos construye el grafo 
    como objeto Data de Pytorch Geometric y lo normaliza'''

    node_feats = ['r', 'theta', 'z', 'layer'] 
    edge_feats = ['delta_eta', 'delta_phi']

    hits_in = segments[['hit_id_in'] + [f"{f}_in" for f in node_feats]].rename(columns={**{f"{f}_in": f for f in node_feats}, 'hit_id_in': 'hit_id'})
    hits_out = segments[['hit_id_out'] + [f"{f}_out" for f in node_feats]].rename(columns={**{f"{f}_out": f for f in node_feats}, 'hit_id_out': 'hit_id'})
    
    hits_all = pd.concat([hits_in, hits_out], ignore_index=True) #verticalmente
    hits_all = hits_all.drop_duplicates('hit_id').reset_index(drop=True) #porque entre capa y capa cada hit parece dos veces
    hits_all = hits_all.sort_values('r').reset_index(drop=True)
    hits_all['node_idx'] = hits_all.index
    
    #matriz x de nodos [num_nodos, num_node_feats]
    x = torch.tensor(hits_all[['r','theta','z']].values, dtype=torch.float) # estos son los nodos del grafo

    #Probamos si ayuda meter como input de los ndoos la layer 
    layer_feat = torch.tensor(hits_all['layer'].values, dtype=torch.float).unsqueeze(1)
    x = torch.tensor(hits_all[['r','theta','z']].values, dtype=torch.float)
    x = torch.cat([x, layer_feat], dim=1)

    id2idx = dict(zip(hits_all['hit_id'], hits_all['node_idx']))

    #aristas edge_index: dos filas [origenes; destinos]
    src = segments['hit_id_in'].map(id2idx).values
    dst = segments['hit_id_out'].map(id2idx).values
    
    graph = Data(
        x=x,
        edge_index=torch.tensor(np.array([src, dst]), dtype=torch.long), # matriz (2, num_edges) donde 2 es el idx de origen y el de destino
        edge_attr=torch.tensor(segments[edge_feats].values, dtype=torch.float),
        hit_id = torch.tensor(hits_all['hit_id'].values, dtype=torch.long),
        layer=torch.tensor(hits_all['layer'].astype(int).values, dtype=torch.long),
        y=torch.tensor(segments['label'].astype(int).values, dtype=torch.long))
    
    if drop_y:
        delattr(graph, 'y') #eliminamos la etiqueta para no usarla en la predicción

    graph = utils.normalize_graph(graph)
    return graph

def process_event_batches(n_event, drop_y, get_true_tracks, dtheta, dphi, pt_min = 0.5, pt_max = 1000, path = 'data/events'):
    
    '''Combina las funciones anteriores para resumir la creación de datos en una sola función'''

    hits = pd.read_csv(f'{path}/event{n_event}-hits.csv')
    particles = pd.read_csv(f'{path}/event{n_event}-particles.csv')
    truth = pd.read_csv(f'{path}/event{n_event}-truth.csv')

    minibatches = space_batches(hits, truth, particles, n_event, dtheta, dphi)
    graphs = []
    for minibatch in minibatches:
        graphs.append(graph_builder(get_segments_batches(minibatch, pt_min, pt_max), drop_y))

    if get_true_tracks: # para metricas obtenemos las trayectorias reales del evento
        #siempre con filtrado ideal de momentos porque no forma parte de la reconstruccion
        particles['pt'] = np.sqrt(particles.px**2 + particles.py**2)
        particles_filt = particles[(particles['pt'] > 0.5) & (particles['pt'] < 1000)]

        merged = pd.merge(truth[['hit_id', 'particle_id']], particles_filt[['particle_id', 'pt', 'q']], on='particle_id', how='inner')

        barrels = [(8,2), (8,4), (8,6), (8,8), (13,2), (13,4), (13,6), (13,8), (17,2), (17,4)]
        groups = hits.groupby(['volume_id', 'layer_id'])
        hits = pd.concat([groups.get_group(barrels[i]).assign(layer=i) for i in range(len(barrels))])
        hits['r'] = np.sqrt(hits.x**2 + hits.y**2)
        hits['theta'] = np.arctan2(hits.y, hits.x)
        hits['phi'] = np.arccos(hits.z / np.sqrt(hits.r**2 + hits.z**2))

        hits = hits[(hits['phi'] >= azimutal_cut) & (hits['phi'] <= np.pi - azimutal_cut)]    #mismo corte angular

        df = pd.merge(merged, hits[['hit_id', 'r', 'theta', 'z', 'layer']], on='hit_id', how='inner') #juntamos toda la info

        unique_particles = sorted(df['particle_id'].unique()) # track_id únicos a cada particle_id
        mapping = {pid: idx + 1 for idx, pid in enumerate(unique_particles)}
        df['track_id'] = df['particle_id'].map(mapping)

        #filtro de inicio en capas 0, 1 o 2
        initial_ids = set(df['track_id'].unique())
        valid_start_ids = df[df.layer.isin([0, 1, 2])]['track_id'].unique()
        df = df[df.track_id.isin(valid_start_ids)]
        removed_start = initial_ids - set(valid_start_ids)
        print(f"Tracks descartados por no empezar en capas 0, 1 o 2: {len(removed_start)}")

        #filtro de capas consecutivas
        def is_consec(layers):
            L = sorted(set(layers))
            return L == list(range(L[0], L[-1] + 1))

        consec_ids = [tid for tid, g in df.groupby('track_id') if is_consec(g['layer'])]
        before_consec = set(df.track_id.unique())
        df = df[df.track_id.isin(consec_ids)]
        removed_consec = before_consec - set(consec_ids)
        print(f"Tracks descartados por capas no consecutivas: {len(removed_consec)}")

        #maximo un hit por capa
        df = df.sort_values(by=['track_id', 'layer', 'r'])
        df = df.drop_duplicates(subset=['track_id', 'layer'], keep='first')

        # filtro longitud entre 7 y 10 hits
        track_counts = df.track_id.value_counts()
        valid_len_ids = track_counts[(track_counts >= 7) & (track_counts <= 10)].index
        before_len = set(df.track_id.unique())
        df = df[df.track_id.isin(valid_len_ids)]
        removed_len = before_len - set(valid_len_ids)
        print(f"Tracks descartados por no tener entre 7 y 10 hits: {len(removed_len)}") 
       
        new_ids = {tid: i+1 for i, tid in enumerate(sorted(df['track_id'].unique()))}  #renumerar
        df['track_id'] = df['track_id'].map(new_ids)

        result = df[['hit_id', 'r', 'theta', 'z', 'track_id', 'pt', 'q']].copy()
        ground_tracks = result.sort_values('track_id').reset_index(drop=True)

    
    else:
        ground_tracks = None

    
    return graphs, ground_tracks