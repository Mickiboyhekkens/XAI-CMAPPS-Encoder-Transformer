import numpy as np
import pandas as pd
import matplotlib . pyplot as plt
import math
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from helper_functions import *
import seaborn as sb 
# import other libraries

def load_data(path, column_names):
    df = pd.read_csv(path, index_col=None, delim_whitespace=True, header=None, names=column_names)
    return df

def plot_all_sensors(df, engine_id, sensor_ids, title='plot'):
    # Number of subfigures, define the grid layout
    n_subfigures = len(sensor_ids)
    rows = 7
    cols = math.ceil(n_subfigures/7)

    df_engine = df[df['unit number'] == engine_id]
    time = df_engine['cycle'].to_numpy()
    
    # Create the figure and axes objects
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 20))
    
    # Loop over all possible subplots within the grid
    for i in range(n_subfigures):
        ax = axes[i // cols, i % cols]  # Locate the right subplot
        if i < n_subfigures:

            sensor_data = df_engine[f'sensor measurement {sensor_ids[i]}'].to_numpy()
            
            ax.scatter(time, sensor_data, s=1)
            ax.set_xlabel('t (cycles)')
            ax.set_title(f"Sensor {sensor_ids[i]}")
        else:
            ax.axis('off')  # Turn off axis for unused plots
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_operational_conditions(df, engine_id):
    # Filter the DataFrame for the specified engine ID
    df_engine = df[df['unit number'] == engine_id]

    # Extract the operational parameters
    altitude = df_engine['altitude'].to_numpy()
    tra = df_engine['TRA'].to_numpy()
    mach_nr = df_engine['mach_nr'].to_numpy()

    # Create the figure and 3D axes objects
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a scatter plot in 3D space without a color map
    ax.scatter(altitude, tra, mach_nr, color='b', marker='o', s=15)  # 'b' stands for blue

    # Labeling the axes
    ax.set_xlabel('Altitude')
    ax.set_ylabel('TRA')
    ax.set_zlabel('Mach Number')

    # Set a title for the plot')
    
    # Adjust layout to ensure labels are not cut off
    plt.subplots_adjust(left=0.1, right=2, top=0.9, bottom=0.1)
    
    # Show the plot
    plt.show()


def get_operational_clusters(df, dataset_id):
    if dataset_id == 2 or dataset_id == 4:
        op_condts = df[['altitude', 'TRA', 'mach_nr']]
        op_condts.shape
        # Use K-means to find clusters
        kmeans = KMeans(n_clusters = 6) # Greedy k-means is used when the n_clusters is not specified, it enables faster convergence
        kmeans.fit(op_condts)
        cluster_labels = kmeans.labels_
    else:
        cluster_labels = np.zeros(df.shape[0])
    return cluster_labels

def add_rul(g):
    g['RUL'] = max(g['cycle'])  - g['cycle']
    return g

def add_FLAT(g, dataset_id):
    if dataset_id == 1 or dataset_id == 3:
        if max(g['sensor measurement 10'])  - min(g['sensor measurement 10']) < 0.000001:
            g['FLAT'] = 1
        else:
            g['FLAT'] = 0
        return g
    if dataset_id == 2 or dataset_id == 4:
        g['FLAT'] = 1
        for i in range(6):
            g_op = g[g['op cluster'] == i]
            if max(g_op['sensor measurement 10']) - min(g_op['sensor measurement 10']) < 0.000001:
                g['FLAT'] = 0
        return g

def plot_pearson_correlation(df):
    # plotting correlation heatmap 
    #df_engine = df[df['unit number'] == engine_id]
    df_pearson = df.iloc[:, 5:-2]
    dataplot=sb.heatmap(df_pearson.corr()) 
    # displaying heatmap 
    plt.show()

def drop_sensors(df_to_drop, ids):
    setting_names = ['altitude', 'TRA', 'mach_nr']
    
    if ids == 1:
        drop_ids= [1, 5, 6, 10, 16, 18, 19]
    if ids == 2:
        drop_ids = [1, 5, 6, 10, 16, 18, 19]
    if ids == 3:
        drop_ids = [1, 5, 6, 16, 18, 19]
    if ids == 4:
        drop_ids = [1, 5, 6, 10, 16, 18, 19]
   
    drop_names = [f'sensor measurement {id}' for id in drop_ids]
    drop_labels = setting_names + drop_names 
    df_to_drop.drop(labels=drop_labels, axis=1, inplace=True)
    
    return df_to_drop, drop_ids

def normalize_min_max(df, sensor_names):
    df_new = df.copy()
    min_values = {}
    max_values = {}
    
    # Group the DataFrame by 'unit number'
    for name, group in df.groupby('unit number'):
        # Store min and max for each group
        min_values[name] = group[sensor_names].min()
        max_values[name] = group[sensor_names].max()
        # Normalize each group and update the DataFrame
        df_new.loc[group.index, sensor_names] = (group[sensor_names] - min_values[name]) / (max_values[name] - min_values[name])
    
    return df_new, min_values, max_values

def normalize_standard(df, sensor_names):
    df_new = df.copy()
    mean_values = {}
    std_values = {}
    
    # Group the DataFrame by 'unit number'
    for name, group in df.groupby('unit number'):
        # Store mean and std for each group
        mean_values[name] = group[sensor_names].mean()
        std_values[name] = group[sensor_names].std()
        # Normalize each group and update the DataFrame
        df_new.loc[group.index, sensor_names] = (group[sensor_names] - mean_values[name]) / std_values[name]
    
    return df_new, mean_values, std_values

def ewma(data, rho):
    ewma_bias_corr = np.empty(0)
    s_prev = 0
    rho = rho

    for i, y in enumerate(data):   
        # Variables to store smoothed data point
        s_cur = 0
        s_cur_bc = 0  
        s_cur = rho * s_prev + (1 - rho) * y
        s_cur_bc = s_cur / ((1 - rho**(i + 1)))
        # Append new smoothed value to array
        ewma_bias_corr = np.append(ewma_bias_corr, s_cur_bc)   
        s_prev = s_cur
    return ewma_bias_corr

def ewma_sensors(df_input, sensor_names):
    df_new = df_input.copy()
    
    grouped = df_new.groupby('unit number')

    df_ewma = pd.DataFrame()
    
    for engine, engine_df in grouped:
        for name in sensor_names:
            
            data = engine_df[name]  # Get data for the current sensor and engine
            ewma_sensor = ewma(data, 0.90)  # Compute EWMA for the current sensor data
            engine_df[name] = ewma_sensor
        df_ewma = pd.concat([df_ewma, engine_df],  ignore_index=True)

    return df_ewma

def standard_norm(df, sensor_names):
    title = df.iloc[:, 0:2]
    data = df[sensor_names]
    data_norm = (data-data.mean())/data.std()  # min-max normalization
    # data_norm = (data-data.mean())/data.std()  # standard normalization (optional)
    df_norm = pd.concat([title, data_norm], axis=1)
    return df_norm

def min_max_norm(df, sensor_names):
    title = df.iloc[:, 0:2]
    data = df[sensor_names]
    data_norm = (data - data.min()) / (data.max() - data.min())  # min-max normalization
    # data_norm = (data-data.mean())/data.std()  # standard normalization (optional)
    df_norm = pd.concat([title, data_norm], axis=1)
    return df_norm

def plot_embedding_3d(X_tsne):
# Visualize the embedding in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c='b', marker='o', s=0.01)  # Adjust marker, color, and size as needed
    ax.set_title('t-SNE 3D Visualization')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.show()

def embed(X, num_dimensions):
    tsne = TSNE(n_components=num_dimensions, random_state=42)
    X_tsne = tsne.fit_transform(X)
    return X_tsne


def plot_clusters_RUL(rul, ewma_TSNE, interval_list, num_dimensions):
    # Initialize clusters with one cluster for each interval boundary and one extra for values above the highest boundary
    clusters = {}
    
    for i in range(len(interval_list)-1):
        interval = f'{interval_list[i]} - {interval_list[i+1]}'
        clusters[interval] = []

    # Group RUL values into clusters based on interval_list
    for i, RUL in enumerate(rul):
        if RUL > interval_list[0] and RUL < interval_list[-1]:
            interval_index = 1
            
            while interval_index < len(interval_list) and RUL > interval_list[interval_index]:
                interval_index += 1
            interval = f'{interval_list[interval_index-1]} - {interval_list[interval_index]}'
            clusters[interval].append(ewma_TSNE[i])

    print(clusters)

    # Visualize the embedding with colored clusters
    fig = plt.figure(figsize=(3, 3))
    if num_dimensions == 2:
        ax = fig.add_subplot(111)
    elif num_dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        print("Invalid number of dimensions. Please choose 2 or 3.")
        return

    # Define colors for each cluster except the first one
    colors = plt.cm.jet(np.linspace(0, 1, len(clusters)))

    for i, interval in enumerate(clusters):
        if num_dimensions == 2:
            ax.scatter(*zip(*clusters[interval]), color=colors[i], label=f'RUL {interval}', s=0.1)
        elif num_dimensions == 3:
            ax.scatter(*zip(*clusters[interval]), color=colors[i], label=interval, s=0.1)

    # Setup axis labels and title
    if num_dimensions == 2:
        ax.set_title(f'RUL Clusters')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
    elif num_dimensions == 3:
        ax.set_title(f'RUL Clusters')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')

    # Add a legend to the right side of the plot
    #ax.legend(loc='center left', bbox_to_anchor=(0.14, -0.4), markerscale=20)

    plt.show()

def plot_clusters_kmeans(X_tsne, cluster_labels, title):
    # Find unique clusters
    clusters = np.unique(cluster_labels)
    clusters_dict = {}
    
    # Initialize dictionary to hold embeddings for each cluster
    for cluster in clusters:
        clusters_dict[cluster] = []
    
    # Populate the dictionary with embeddings
    for i, embedding in enumerate(X_tsne):
        label = cluster_labels[i]
        clusters_dict[label].append(embedding)
    
    # Convert lists to numpy arrays for easier slicing
    for cluster in clusters:
        clusters_dict[cluster] = np.array(clusters_dict[cluster])
    
    # Plot settings
    plt.figure(figsize=(3, 3))
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'orange', 'purple', 'brown']  # More colors can be added if needed

    # Plot each cluster with a different color and marker
    for i, cluster in enumerate(clusters):
        plt.scatter(clusters_dict[cluster][:, 0], clusters_dict[cluster][:, 1], 
                    color=colors[i % len(colors)],  
                    s=0.01,  # Adjust size as needed
                    label=f'Cluster {cluster}')
    
    # Legend and show plot
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.show()

