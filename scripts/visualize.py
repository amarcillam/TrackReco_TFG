import numpy as np
import plotly.graph_objects as go
import pandas as pd

def plot_tracks_compare(df_ground, df_pred, title='Tracks True reconstruidos: Ground vs Prediction'):

    '''plot de los tracks reconstruidos finales y de los tracks solución en 3d'''

    for df in (df_ground, df_pred):
        df['x'] = df['r'] * np.cos(df['theta'])
        df['y'] = df['r'] * np.sin(df['theta'])

    fig = go.Figure()

    colors = ['blue', 'red']
    labels = ['Ground Truth', 'Prediction']

    for df, color, label in zip([df_ground, df_pred], colors, labels):
        unique_tracks = list(df['track_id'].unique())
        for i, (tid, group) in enumerate(df.groupby('track_id')):
            if tid == -1:
                continue
            group = group.sort_values('z')
            fig.add_trace(go.Scatter3d(
                x=group['x'],
                y=group['y'],
                z=group['z'],
                mode='lines',
                line=dict(color=color, width=1),
                showlegend=bool(i == 0),
                name=label if i == 0 else None
            ))

    #origen
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(color='red', size=3),
        name='Origen (0,0,0)',
        showlegend=True
    ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X [mm]', yaxis_title='Y [mm]', zaxis_title='Z [mm]'),
        width=900,
        height=700,
    )
    fig.show()

def plot_graph_trajectories(graph, title = 'Segmentos del grafo True/False'):

    '''plot de los segmentos True y False'''

    coords = graph.x.cpu().numpy()
    r = coords[:, 0]
    theta = coords[:, 1]
    z = coords[:, 2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    edge_idx = graph.edge_index.cpu().numpy()
    edge_labels = graph.y.cpu().numpy()

    #listas de puntos para cada color
    xb, yb, zb = [], [], []  #y = 1 azul
    xr, yr, zr = [], [], []  #y =0 rojo 

    for i in range(edge_idx.shape[1]):
        src = edge_idx[0, i]
        dst = edge_idx[1, i]
        x0, y0, z0 = x[src], y[src], z[src]
        x1, y1, z1 = x[dst], y[dst], z[dst]
        
        if edge_labels[i] == 1:
            xb += [x0, x1, None]
            yb += [y0, y1, None]
            zb += [z0, z1, None]
        else:
            xr += [x0, x1, None]
            yr += [y0, y1, None]
            zr += [z0, z1, None]

    #lineas entre puntos
    trace_blue = go.Scatter3d(
        x=xb, y=yb, z=zb,
        mode='lines',
        line=dict(color='blue', width=1),
        name='y = 1'
    )
    trace_red = go.Scatter3d(
        x=xr, y=yr, z=zr,
        mode='lines',
        line=dict(color='rgba(255,0,0,0.5)', width=1),  #rojo semitransparente
        name='y = 0'
    )

    fig = go.Figure(data=[trace_blue, trace_red])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X [mm]', yaxis_title='Y [mm]', zaxis_title='Z [mm]'),
        width=900,
        height=700,
    )
    fig.show()

def stats(pred_df, ground_df, n_matched_hits):

    '''Cálculo de la eficiencia y pureza comparando los tracks reconstruidos y los reales'''

    n_hits_min = 7
    #filtro tracks cortos
    gf = ground_df.groupby('track_id').filter(lambda df: len(df) >= n_hits_min)
    pf = pred_df.groupby('track_id').filter(lambda df: len(df) >= n_hits_min)

    m = (pf[['hit_id','track_id']].rename(columns={'track_id':'pred'}).merge(gf[['hit_id','track_id']].rename(columns={'track_id':'gt'}),on='hit_id'))

    if m.empty:
        print("Purity: 0.00%  Efficiency: 0.00%")
        return pd.DataFrame(columns=pred_df.columns), None
    
    c = m.groupby(['pred','gt']).size().reset_index(name='n') #contamos hits compartidos
    #metricas eficiencia y pureza
    tp = (c.groupby('pred').n.max() >= n_matched_hits).sum()
    fg = (c.groupby('gt'  ).n.max() >= n_matched_hits).sum()
    P  = tp / pf['track_id'].nunique() * 100
    E  = fg / gf['track_id'].nunique() * 100
    
    print(f'Reconstrucción de tracks con minimo {n_hits_min} hits y matches de {n_matched_hits} o más') #los denominadores contienen tambien los tracks ground truth con mas n_hits al menos
    print(f"Purity: {P:.2f}% ({tp}/{pf['track_id'].nunique()})")
    print(f"Efficiency: {E:.2f}% ({fg}/{gf['track_id'].nunique()})")
    return P, E
