import time
import random
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, mean_squared_error
from sklearn.model_selection import GridSearchCV 
from hypervem import Hypergraph
from hypervem import HypergraphCoClustering
from hypervem.utils import match_labels, relabel
from hypervem.visualizations import (
    visualize_parameters,
    plot_hypergraph,
    plot_sorted_hypergraph,
    plot_model_selection_heatmap
)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def run_synthetic_demo(
    nodes: int = 200,
    hyperedges: int = 200,
    n_node_clusters_true: int = 2,
    n_hyperedge_clusters_true: int = 2,
    seed: int = 42,
    run_model_selection: bool = True,
    show_plots: bool = True,
    verbose = False
):
    """
    Runs a full synthetic experiment:
    - data generation
    - VEM fitting
    - (optional) model selection using sklearn GridSearchCV
    - clustering & parameter recovery evaluation
    - (optional) plotting

    Intended to be imported and used from notebooks or main.py.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    hg = Hypergraph.generate_random(
        num_hyperedges=hyperedges,
        num_nodes=nodes,
        total_node_groups=n_node_clusters_true,
        total_hyperedge_groups=n_hyperedge_clusters_true,
    )
    
    data_time = time.time() - start_time

    if run_model_selection:
        # Generate the candidate K, G values
        offsets = [-3, -2, -1, 0, 1, 2, 3]
        K_cand = [n_node_clusters_true + i for i in offsets if n_node_clusters_true + i >= 2]
        G_cand = [n_hyperedge_clusters_true + i for i in offsets if n_hyperedge_clusters_true + i >= 2]

        
        param_grid = {
            'n_node_clusters': K_cand,
            'n_hyperedge_clusters': G_cand
        }

        # Fit on full dataset.
        # ICL is an information criterion, not a hold-out metric.
        indices = np.arange(hg.num_hyperedges)
        custom_cv = [(indices, indices)]

        print(f"Running GridSearchCV on {len(K_cand) * len(G_cand)} candidates...")
        
        # Initialize the base estimator
        base_vem = HypergraphCoClustering(
            max_iter=100,
            init_method="kmeans",
            verbose=verbose, 
            n_init=3,
            random_state=seed
        )

        grid_search = GridSearchCV(
            estimator=base_vem,
            param_grid=param_grid,
            scoring=None,
            cv=custom_cv,
            verbose=1,
            n_jobs=-1
        )
        
       
        grid_search.fit(hg.incidence_matrix)
        
        vem = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_} with ICL: {grid_search.best_score_:.2f}")
       
    else:
        # Run a single VEM fit if selection is off
        vem = HypergraphCoClustering(
            K=n_node_clusters_true,
            G=n_hyperedge_clusters_true,
            max_iter=100,
            init_method="kmeans",
            verbose=verbose,
            n_init=3,
        )
        
        vem.fit(hg.incidence_matrix)
    
    # Retrieve clusters and compare them to the ground truth
    true_u = hg.ground_truth["z_nodes"]
    true_w = hg.ground_truth["z_hyperedges"]

    pred_u = vem.node_clusters_
    pred_w = vem.hyperedge_clusters_

    ari_nodes = adjusted_rand_score(true_u, pred_u)
    ari_edges = adjusted_rand_score(true_w, pred_w)

    # Compare recovery of the parameters
    # This also requires the alignment of the labels, which is done
    # by match_labels() -> relabel()
    
    if vem.n_node_clusters == n_node_clusters_true and vem.n_hyperedge_clusters == n_hyperedge_clusters_true:
        perm_k = match_labels(true_u, pred_u)
        perm_g = match_labels(true_w, pred_w)

        pred_u_aligned = relabel(pred_u, perm_k)
        pred_w_aligned = relabel(pred_w, perm_g)

        est_gamma = vem.gamma_[perm_k]
        est_delta = vem.delta_[perm_g]
        est_theta = vem.theta_[perm_k][:, perm_g]

        mse_gamma = mean_squared_error(hg.ground_truth["gamma"], est_gamma)
        mse_delta = mean_squared_error(hg.ground_truth["delta"], est_delta)
        mse_theta = mean_squared_error(hg.ground_truth["theta"].flatten(), est_theta.flatten())
         
    else:
        print(f"Skipping Parameter MSE: Model selection found K={vem._K}, G={vem._G} (True: K={n_node_clusters_true}, G={n_hyperedge_clusters_true})")
        mse_gamma, mse_delta, mse_theta = np.nan, np.nan, np.nan
        pred_u_aligned = pred_u  # Fallback for crosstab
        pred_w_aligned = pred_w


    # Parameter estimation tables 
    summary = pd.DataFrame(
        {
            "Metric": [
                "ARI (nodes)",
                "ARI (hyperedges)",
                "MSE (gamma)",
                "MSE (delta)",
                "MSE (theta)",
            ],
            "Value": [
                ari_nodes,
                ari_edges,
                mse_gamma,
                mse_delta,
                mse_theta,
            ],
        }
    )

    print(summary.to_string(index=False))

    # Clustering accuracy tables
    node_crosstab = pd.crosstab(
        true_u,
        pred_u_aligned,
        rownames=["True node cluster"],
        colnames=["Predicted node cluster"],
    )

    edge_crosstab = pd.crosstab(
        true_w,
        pred_w_aligned,
        rownames=["True hyperedge cluster"],
        colnames=["Predicted hyperedge cluster"],
    )

    print(node_crosstab)
    print(edge_crosstab)

    # Plots
    if show_plots:    
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)

        gs = GridSpec(
            nrows=2,
            ncols=2,
            figure=fig,
            height_ratios=[1, 0.8],   # bottom much larger
            width_ratios=[1.2, 1.2]   # matrices wider than diagnostics
        )

        ax1 = fig.add_subplot(gs[0, 0])  # ICL heatmap
        ax2 = fig.add_subplot(gs[0, 1])  # parameter diagram
        ax3 = fig.add_subplot(gs[1, 0])  # raw hypergraph matrix
        ax4 = fig.add_subplot(gs[1, 1])  # clustered matrix

        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df.rename(columns={'param_n_node_clusters': 'K', 'param_n_hyperedge_clusters': 'G', 'mean_test_score': 'icl'})
        results_dict = results_df[['K', 'G', 'icl']].to_dict(orient='records')

        plot_model_selection_heatmap(
            results_dict, 
            show_legend=False, 
            ax=ax1
        )

        visualize_parameters(
            vem.gamma_,
            vem.delta_,
            vem.theta_,
            display_connectivity=True,
            ax = ax2
        )

        plot_hypergraph(
            vem.X_,
            yes_color="black", 
            no_color="white", 
            plot_legend=False, 
            ax=ax3
        )
        plot_sorted_hypergraph(
            vem.X_,
            pred_w,
            pred_u,
            yes_color="black",
            no_color="white",
            use_legend=True,
            line_color = "red",
            show_group_labels= False,
            ax = ax4
        )
        for txt in ax2.texts:
            if "Node" in txt.get_text() or "Hyperedge" in txt.get_text():
                txt.set_fontsize(9)
        
        ax2.set_title("Co-clustering parameters \n (line thickness proportional to connection prob.)")

        ax1.set_ylabel(ax1.get_ylabel(), fontsize=10)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=10)

        ax3.set_title("Hypergraph incidence matrix (original)", fontsize=12)
        ax3.set_ylabel("Hyperedges", fontsize = 10)
        ax3.set_xlabel("Nodes", fontsize = 10)

        ax4.set_title("Hypergraph incidence matrix (co-clustered)", fontsize=12)
        ax4.set_ylabel("Hyperedges", fontsize = 10)
        ax4.set_xlabel("Nodes", fontsize = 10)
        plt.show()

    # Return everything useful
    return {
        "vem": vem,
        "hypergraph": hg,
        "ari_nodes": ari_nodes,
        "ari_edges": ari_edges,
        "mse_gamma": mse_gamma,
        "mse_delta": mse_delta,
        "mse_theta": mse_theta,
        "data_time": data_time,
    }