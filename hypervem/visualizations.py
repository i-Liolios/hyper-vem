import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import sparse
import warnings
import matplotlib.gridspec as gridspec

def _plot_dense_matrix(
    X: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    yes_color: str,
    no_color: str,
    nan_color: str,
    yes_label: str,
    no_label: str,
    nan_label: str,
    use_legend: bool = True,
    tick_labels: dict | None = None,
    boundaries: dict | None = None,
    line_color: str = "black",
    show_ticks: bool = False,
    label_fontsize: int = 10,
    ax = None, 
    figsize: tuple = (12, 6)        
):                     
    """
    Internal helper for plotting the hypergraph matrix.
    
    Handles the low-level matplotlib calls (imshow, text placement, 
    patches, legend).

    Parameters
    ----------
    X : np.ndarray
        The dense matrix to plot.
    
    tick_labels : dict, optional
        Custom placement for labels. Expected structure:
        {'rows': {label_str: y_coord, ...}, 
         'cols': {label_str: x_coord, ...}}
    
    boundaries : dict, optional
        Locations to draw separator lines. Expected structure:
        {'rows': [y_indices...], 'cols': [x_indices...]}
    
    ax : plt.Axes, optional
        The axis to plot on. If None, creates a new figure using `figsize`.

    Returns
    -------
    plt.Axes
        The axis with the plot attached.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    cmap = ListedColormap([no_color, yes_color])
    cmap.set_bad(color=nan_color)

    ax.imshow(X, aspect='auto', cmap=cmap, interpolation='none')

    # Draw boundaries if provided
    if boundaries:
        for y in boundaries.get('rows', []):
            ax.axhline(y + 0.5, color=line_color, linewidth=2.2, alpha=1)
        for x in boundaries.get('cols', []):
            ax.axvline(x + 0.5, color=line_color, linewidth=2.2, alpha=1)

    # Add custom text labels
    if tick_labels:
        # Row labels (left side)
        for label, y in tick_labels.get('rows', {}).items():
            ax.text(
                -1.5, y, str(label),
                va="center", ha="right",
                color="black",
                fontsize=label_fontsize,
                rotation=0
            )
        # Column labels (top side)
        for label, x in tick_labels.get('cols', {}).items():
            ax.text(
                x, 1.05 * X.shape[0], str(label),
                va="bottom", ha="center",
                color="black",
                fontsize=label_fontsize,
                rotation=0
            )
        
    # Setters instead of global plt functions
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=25 if tick_labels else None)
    ax.set_ylabel(ylabel, fontsize=12 if tick_labels else None)
    ax.yaxis.set_label_position("right" if tick_labels else "left")
    if use_legend:
        if np.issubdtype(X.dtype, np.floating):
            has_nans = np.isnan(X).any()
        else:
            has_nans = False

        legend_elements = [
            Patch(facecolor=yes_color, label=yes_label, edgecolor="black"),
            Patch(facecolor=no_color, label=no_label, edgecolor="black"),
        ]
        
        # Only add the NaN patch if NaNs are detected
        if has_nans:
            legend_elements.append(
                Patch(facecolor=nan_color, edgecolor='black', label=nan_label)
            )

        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax

def _check_and_convert_sparse(X: np.ndarray | sparse.csr_array, force_dense: bool) -> np.ndarray:
    """
    Helper to safely check and convert sparse matrices to dense.
    """
    if sparse.issparse(X): 
        if not force_dense:
            warnings.warn(f"Sparse matrix detected. Use force_dense=True to transform into dense.")
            return None
        return X.toarray()
    return X


def plot_hypergraph(
    X: np.ndarray | sparse.csr_array,
    xlabel: str = "Nodes",
    ylabel: str = "Hyperedges",
    title: str = "Hypergraph Matrix",
    yes_color: str = '#377eb8',
    no_color: str = '#ff7f00',
    nan_color: str = "#ffffff",
    plot_legend: bool = True,
    yes_label: str = "1",
    no_label: str = "0",
    nan_label: str = "NaN",
    force_dense: bool = False,
    show_ticks: bool = False,
    label_fontsize: int = 10,
    ax = None,
    figsize: tuple = (12, 6)      
):
    """
    Plots the raw incidence matrix of a hypergraph.

    Visualizes the connection between nodes (rows) and hyperedges (columns).

    Parameters
    ----------
    X : np.ndarray or sparse.csr_array
        The incidence matrix of shape (n_nodes, n_hyperedges).

    xlabel : str, default="Nodes"
        Label for the x-axis.

    ylabel : str, default="Hyperedges"
        Label for the y-axis.

    title : str, default="Hypergraph Matrix"
        Title of the plot.

    yes_color : str, default='#377eb8'
        Color for present entries (1s).

    no_color : str, default='#ff7f00'
        Color for absent entries (0s).

    nan_color : str, default="#ffffff"
        Color for NaN entries (if any).

    plot_legend : bool, default=True
        Whether to show the legend explaining the colors.

    force_dense : bool, default=False
        If True, converts sparse matrices to dense arrays for plotting. 

    show_ticks : bool, default=False
        If True, shows numeric ticks on axes.

    ax : plt.Axes, optional
        Matplotlib axes object. If None, creates a new figure.

    Returns
    -------
    plt.Axes or None
        The axes object, or None if sparse conversion failed (and force_dense=False).
    """
    X_dense = _check_and_convert_sparse(X, force_dense)
    if X_dense is None:
        return None

    return _plot_dense_matrix(
        X=X_dense,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        yes_color=yes_color,
        no_color=no_color,
        nan_color=nan_color,
        yes_label=yes_label,
        no_label=no_label,
        nan_label=nan_label,
        use_legend=plot_legend,
        show_ticks=show_ticks,
        label_fontsize=label_fontsize,
        ax=ax,
        figsize=figsize
    )

def plot_sorted_hypergraph(
    X: np.ndarray | sparse.csr_array, 
    row_labels: np.ndarray, 
    col_labels: np.ndarray, 
    title: str = "Clustered Incidence Matrix", 
    xlabel: str = "Nodes (Grouped)", 
    ylabel: str = "Hyperedges (Grouped)", 
    yes_color: str = '#377eb8',
    no_color: str = '#ff7f00',
    nan_color: str = "#ffffff",
    use_legend: bool = True,
    yes_label: str = "1",
    no_label: str = "0",
    nan_label: str = "NaN",
    figsize: tuple = (12, 6),
    show_boundaries: bool = True,
    line_color: str = "black",
    force_dense: bool = False,
    show_group_labels: bool = True,
    label_fontsize: int = 10,
    custom_group_rows: list | None = None,
    custom_group_cols: list | None = None,
    ax = None 
):
    """
    Sorts and plots a hypergraph incidence matrix based on cluster assignments.

    Rows (nodes) and columns (hyperedges) are reordered so that members of the 
    same cluster appear together. Boundaries are drawn between clusters.

    Parameters
    ----------
    X : np.ndarray or sparse.csr_array
        The incidence matrix of shape (n_nodes, n_hyperedges).

    row_labels : np.ndarray
        Cluster assignments for nodes. Shape must be (n_nodes,).

    col_labels : np.ndarray
        Cluster assignments for hyperedges. Shape must be (n_hyperedges,).

    title : str, default="Clustered Incidence Matrix"
        Title of the plot.

    show_boundaries : bool, default=True
        If True, draws lines separating the clusters.

    show_group_labels : bool, default=True
        If True, adds text labels for the cluster IDs on the axes.

    custom_group_rows : list, optional
        List of strings to replace numeric row cluster IDs (e.g., ['Group A', 'Group B']).

    custom_group_cols : list, optional
        List of strings to replace numeric column cluster IDs.

    force_dense : bool, default=False
        If True, forces conversion of sparse input to dense.

    ax : plt.Axes, optional
        Matplotlib axes object. If None, creates a new figure.

    Returns
    -------
    plt.Axes or None
        The axes object, or None if sparse conversion failed.
    """
    X_dense = _check_and_convert_sparse(X, force_dense)
    if X_dense is None:
        return None

    # Sort Data
    row_indices = np.argsort(row_labels)
    col_indices = np.argsort(col_labels)
    sorted_row_labels = row_labels[row_indices]
    sorted_col_labels = col_labels[col_indices]
    
    X_sorted = X_dense[row_indices][:, col_indices]

    # Calculate Boundaries & Labels
    boundaries = {}
    tick_labels = None

    if show_boundaries:
        boundaries['rows'] = np.where(np.diff(sorted_row_labels))[0]
        boundaries['cols'] = np.where(np.diff(sorted_col_labels))[0]

    if show_group_labels:
        tick_labels = {}
        def group_midpoints(sorted_lbls):
            unique_labels, start_indices = np.unique(sorted_lbls, return_index=True)
            end_indices = np.append(start_indices[1:], len(sorted_lbls))
            return {
                lbl: (start + end - 1) / 2
                for lbl, start, end in zip(unique_labels, start_indices, end_indices)
            }

        row_mids = group_midpoints(sorted_row_labels)
        col_mids = group_midpoints(sorted_col_labels)

        # Apply custom labels logic
        final_row_labels = {}
        for lbl, pos in row_mids.items():
            lbl_int = int(lbl) 
            if custom_group_rows and lbl_int < len(custom_group_rows):
                txt = str(custom_group_rows[lbl_int])
            else:
                txt = str(lbl_int + 1)
            final_row_labels[txt] = pos

        final_col_labels = {}
        for lbl, pos in col_mids.items():
            lbl_int = int(lbl)
            if custom_group_cols and lbl_int < len(custom_group_cols):
                txt = str(custom_group_cols[lbl_int])
            else:
                txt = str(lbl_int + 1)
            final_col_labels[txt] = pos
            
        tick_labels['rows'] = final_row_labels
        tick_labels['cols'] = final_col_labels

    return _plot_dense_matrix(
        X=X_sorted,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        yes_color=yes_color,
        no_color=no_color,
        nan_color=nan_color,
        yes_label=yes_label,
        no_label=no_label,
        nan_label=nan_label,
        use_legend=use_legend,
        figsize=figsize,
        tick_labels=tick_labels,
        boundaries=boundaries,
        label_fontsize=label_fontsize,
        line_color=line_color,
        ax=ax
    )

def plot_model_selection_heatmap(
    results_dict: dict,
    show_legend: bool = True,
    ax = None
):
    """
    Plots an ICL model selection landscape over (K, G).
    
    Parameters
    ----------
    results_dict : list of dict or dict
        Dictionary containing model search results.
        Expected keys: 'n_node_clusters', 'n_hyperedge_clusters', 'icl' (or 'param_n_node_clusters', 'param_n_hyperedge_clusters', 'mean_test_score').
        Compatible with `GridSearchCV.cv_results_`.

    Returns
    -------
    None
        Displays the heatmap using matplotlib.show().
    """
    df = pd.DataFrame(results_dict)
    rename_map = {'param_n_node_clusters': 'K', 'param_n_hyperedge_clusters': 'G', 'mean_test_score': 'icl'}
    df = df.rename(columns=rename_map)
    
    required_cols = {'K', 'G', 'icl'}
    if not required_cols.issubset(df.columns):
        print(f"Dataframe missing required columns. Found: {df.columns}")
        return None
    
    # Figure Sizing logic (only used if we create a new figure)
    if ax is None:
        num_k = df['K'].nunique()
        num_g = df['G'].nunique()
        fig_width = min(20, max(8, num_k * 0.8))
        fig_height = min(15, max(6, num_g * 0.6))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    heatmap_data = df.pivot_table(index='G', columns='K', values='icl', aggfunc="mean")

    sns.heatmap(heatmap_data, annot=True, annot_kws={"size": 8}, fmt=".0f", 
                cmap="viridis", cbar = show_legend, cbar_kws={'label': 'ICL Score'}, ax=ax)

    if show_legend:
        if len(ax.collections) > 0:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label('ICL Score', size=10)
    
    ax.invert_yaxis()
    ax.set_title("ICL values for model selection")
    ax.set_xlabel("Number of Node Clusters (K)")
    ax.set_ylabel("Number of Hyperedge Clusters (G)")
    
    return ax

def visualize_parameters(
    node_proportions: np.ndarray, 
    hyperedge_proportions: np.ndarray, 
    theta: np.ndarray, 
    display_connectivity: bool = False,
    node_color: str = 'skyblue',
    hyperedge_color: str = 'lightgreen',
    edge_color: str = 'gray',
    text_color: str = 'black',
    min_theta: float = 0.0,
    ax = None
):
    """
    Visualizes Hypergraph Co-clustering parameters as a bipartite graph.
    
    Node groups (K) are displayed on the left, Hyperedge groups (G) on the right.
    
    Parameters
    ----------
    node_proportions : np.ndarray
        Mixing proportions for node clusters (K,). (gamma)

    hyperedge_proportions : np.ndarray
        Mixing proportions for hyperedge clusters (G,). (delta)

    theta : np.ndarray
        Block connectivity probabilities of shape (K, G).

    display_connectivity : bool, optional (default=False)
        If True, annotates edges with specific connectivity probability values.
        
    node_color : str, optional (default='skyblue')
        Color for node cluster boxes.
        
    hyperedge_color : str, optional (default='lightgreen')
        Color for hyperedge cluster boxes.
        
    edge_color : str, optional (default='gray')
        Color for the connecting lines.
        
    text_color : str, optional (default='black')
        Color for the text labels.
        
    min_theta : float, optional (default=0.0)
        Minimum probability threshold to draw an edge. 
        Increase this (e.g., to 0.05) to hide weak connections and reduce clutter.

    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure and axis are created.

    Returns
    -------
    plt.Axes or None
        The axes containing the plot, or None if the matrix conversion failed.
    """
    
    K = len(node_proportions)
    G = len(hyperedge_proportions)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    left_x = 0.2
    right_x = 0.8 
    
    # Internal Helpers
    def get_centered_y(n, center=0.5, max_span=0.8, min_gap=0.15):
        if n == 1:
            return np.array([center])
        span = min(max_span, (n - 1) * min_gap + 0.2) 
        start = center + span / 2
        end = center - span / 2
        return np.linspace(start, end, n)

    def get_box_size(proportion):
        return 0.05 + (0.08 * proportion)

    node_y = get_centered_y(K)
    hyperedge_y = get_centered_y(G)
    
    # Draw Connectivity (Edges)
    for k in range(K):
        for g in range(G):
            prob = theta[k][g]
            
            if prob <= min_theta:
                continue
            
            lw = 1 + prob * 8      
            alpha = 0.2 + 0.8 * prob 
            
            # Use ax.plot
            ax.plot([left_x, right_x], [node_y[k], hyperedge_y[g]], 
                    color=edge_color, linewidth=lw, alpha=alpha, zorder=1)
            
            if display_connectivity:
                dy = node_y[k] - hyperedge_y[g]
                
                if abs(dy) < 0.1:  
                    t = 0.5        
                elif dy > 0:       
                    t = 0.25       
                else:              
                    t = 0.75       

                lbl_x = left_x + t * (right_x - left_x)
                lbl_y = node_y[k] + t * (hyperedge_y[g] - node_y[k])
                
                ax.text(lbl_x, lbl_y, f"{prob:.2f}", 
                        ha='center', va='center', fontsize=10, color=text_color,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1),
                        zorder=4)

    # Draw Node Groups
    for k in range(K):
        prop = node_proportions[k]
        sz = get_box_size(prop)
        
        # Use ax.add_patch and ax.text
        ax.add_patch(plt.Rectangle(
            (left_x - sz, node_y[k] - sz), 2 * sz, 2 * sz, 
            facecolor=node_color, ec=text_color, zorder=2
        ))
        
        ax.text(left_x, node_y[k], f"{prop:.2f}", 
                ha='center', va='center', fontweight='bold', color=text_color, zorder=3)
        ax.text(left_x - sz - 0.02, node_y[k], f"Node\nGroup {k+1}", 
                ha='right', va='center', fontsize=12, color=text_color)

    # Draw Hyperedge Groups
    for g in range(G):
        prop = hyperedge_proportions[g]
        sz = get_box_size(prop)
        
        ax.add_patch(plt.Rectangle(
            (right_x - sz, hyperedge_y[g] - sz), 2 * sz, 2 * sz, 
            facecolor=hyperedge_color, ec=text_color, zorder=2
        ))
        
        ax.text(right_x, hyperedge_y[g], f"{prop:.2f}", 
                ha='center', va='center', fontweight='bold', color=text_color, zorder=3)
        ax.text(right_x + sz + 0.02, hyperedge_y[g], f"Hyperedge\nGroup {g+1}", 
                ha='left', va='center', fontsize=12, color=text_color)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Co-clustering Parameters", fontsize=14, color=text_color)
    
    return ax

def visualize_parameters_matrix(
    node_proportions: np.ndarray, 
    hyperedge_proportions: np.ndarray, 
    theta: np.ndarray, 
    cmap: str = 'cividis_r',
    show_values: bool = True,
    text_color_threshold: float = 0.5,
    min_bar_size: float = 0.1,
    ax = None
):
    """
    Visualizes Co-clustering parameters as a heatmap matrix.
    Barplots of the Node groups (K) and Hyperedge groups (G) are also
    plotted. 

    Parameters
    ----------
    node_proportions : np.ndarray
        Mixing proportions for node clusters (K,). (gamma)

    hyperedge_proportions : np.ndarray
        Mixing proportions for hyperedge clusters (G,). (delta)

    theta : np.ndarray
        Block connectivity probabilities of shape (K, G).
    
    cmap: str
        Color map for the theta values.
    
    show_values: bool
        If True, explicitely display theta values on the heatmap. 
    
    text_color_threshold : float, default=0.5
        Threshold to switch font color for readability. Values above this 
        threshold are printed in white; values below are printed in black.
    
    min_bar_size : float, default=0.1
        Minimum visual height/width for the bars in the proportion plots. 
        Ensures very small groups remain visible.

    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure and axis are created.

    Returns
    -------
    plt.Axes
        The main axis object containing the heatmap.
    """
    K = len(node_proportions)
    G = len(hyperedge_proportions)
    
    vis_node_props = np.maximum(node_proportions, min_bar_size)
    vis_hyperedge_props = np.maximum(hyperedge_proportions, min_bar_size)

    # If no axis is passed, create one.
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Main Heatmap
    im = ax.imshow(theta, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(G))
    ax.set_yticks(np.arange(K))
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    if show_values:
        for i in range(K):
            for j in range(G):
                val = theta[i, j]
                color = 'white' if val > text_color_threshold else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', 
                        color=color, fontsize=10, fontweight='bold')

    # Top Bar Chart (Hyperedges)
    # bounds = [x, y, width, height] in normalized (0,1) axes coordinates
    ax_top = ax.inset_axes([0, 1.02, 1, 0.15], transform=ax.transAxes)
    ax_top.bar(range(G), vis_hyperedge_props, color='lightgreen', edgecolor='gray', width=0.9)

    ax_top.set_xlim(-0.5, G - 0.5)
    
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['bottom'].set_visible(True)
    ax_top.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    ax_top.set_title("Hyperedge Group Proportions", fontsize=14, pad=25, fontweight='bold')
    
    for i, real_val in enumerate(hyperedge_proportions):
        vis_val = vis_hyperedge_props[i]
        ax_top.text(i, vis_val / 2, f"{real_val:.2f}", ha='center', va='center', fontsize=9, color='black')
        ax_top.text(i, vis_val + 0.02, f"Hyperedge\nGroup {i+1}", 
                    ha='center', va='bottom', fontsize=9, multialignment='center')

    # Left Bar Chart (Nodes)
    # Place to the left (-0.17 x-coord)
    ax_left = ax.inset_axes([-0.17, 0, 0.15, 1], transform=ax.transAxes)
    ax_left.barh(range(K), vis_node_props, color='skyblue', edgecolor='gray', height=0.9)
    
    ax_left.set_ylim(K - 0.5, -0.5) # Invert y-axis to match heatmap
    
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['left'].set_visible(False)
    ax_left.spines['bottom'].set_visible(False)
    ax_left.spines['right'].set_visible(True)
    ax_left.tick_params(bottom=False, labelbottom=False, right=False, left=False, labelleft=False)

    ax_left.set_ylabel("Node Group Proportions", fontsize=14, fontweight='bold', labelpad=25)
    
    for i, real_val in enumerate(node_proportions):
        vis_val = vis_node_props[i]
        ax_left.text(vis_val / 2, i, f"{real_val:.2f}", ha='center', va='center', fontsize=9, color='black')
        ax_left.text(vis_val + 0.04, i + 0.1, f"Node\nGroup {i+1}", 
                     ha='center', va='bottom', rotation=90, 
                     multialignment='center', fontsize=9)

    # Colorbar
    # Place to the right (1.02 x-coord)
    ax_cbar = ax.inset_axes([1.02, 0, 0.05, 1], transform=ax.transAxes)
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(r"Connectivity Probability ($\theta$)", rotation=270, labelpad=15)

    return ax