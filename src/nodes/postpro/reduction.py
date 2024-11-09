import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca_spikes(
    axis,
    spike_instances,
    templates,
    labels,
    detection_status,
    detected_colors,
    missed_colors,
):
    n_spikes = spike_instances.shape[0]

    # get all instances
    instances = np.vstack([spike_instances, templates])

    # Instantiate PCA
    pca_model = PCA(n_components=2)

    # Performs PCA
    pca_model.fit(instances)

    # take first two principal components
    PCA(n_components=2)

    # Perform PCA
    embedding = pca_model.transform(instances)

    # isolate spike embedding
    spike_embedding = embedding[:n_spikes, :]
    labels = labels[:n_spikes]
    detection_status = detection_status[:n_spikes]

    # separate detected and missed data
    scores_for_detected = spike_embedding[detection_status == 1, :]
    scores_for_missed = spike_embedding[detection_status == 0, :]
    cell_labels_for_detected = labels[detection_status == 1]
    cell_labels_for_missed = labels[detection_status == 0]

    # plot reduced detected spikes
    edgecolor = np.tile(np.array([[1], [1], [1]]), len(missed_colors)).T
    visualize_components(
        axis,
        component1=scores_for_detected[:, 0],
        component2=scores_for_detected[:, 1],
        labels=cell_labels_for_detected,
        colors=detected_colors,
        markerfacecolor=detected_colors,
        edgecolor=edgecolor,
    )

    # plot reduced missed spikes
    markerfacecolor = np.tile(np.array([[1], [1], [1]]), len(missed_colors)).T
    visualize_components(
        axis,
        component1=scores_for_missed[:, 0],
        component2=scores_for_missed[:, 1],
        labels=cell_labels_for_missed,
        colors=missed_colors,
        markerfacecolor=markerfacecolor,
        edgecolor=missed_colors,
    )

    # plot fit templates
    templates_embedding = embedding[n_spikes:, :]
    axis.plot(
        templates_embedding[:, 0],
        templates_embedding[:, 1],
        ".r",
        zorder=100,
        markersize=4,
        label="univ. templates",
    )

    plt.legend()
    return embedding


def tsne_spikes(
    axis,
    spike_instances,
    templates,
    labels,
    detection_status,
    detected_colors,
    missed_colors,
):
    n_spikes = spike_instances.shape[0]

    # get all instances
    instances = np.vstack([spike_instances, templates])

    # instantiate tsne
    tsne_model = TSNE(n_components=2, perplexity=70, random_state=2020)

    # perform t-SNE
    embedding = tsne_model.fit_transform(instances)

    # isolate spike embedding
    spike_embedding = embedding[:n_spikes, :]
    labels = labels[:n_spikes]
    detection_status = detection_status[:n_spikes]

    # separate detected and missed data
    embedding_for_detected = spike_embedding[detection_status == 1, :]
    embedding_for_missed = spike_embedding[detection_status == 0, :]
    cell_labels_for_detected = labels[detection_status == 1]
    cell_labels_for_missed = labels[detection_status == 0]

    # plot reduced detected spikes
    edgecolor = np.tile(np.array([[1], [1], [1]]), len(missed_colors)).T
    visualize_components(
        axis,
        component1=embedding_for_detected[:, 0],
        component2=embedding_for_detected[:, 1],
        labels=cell_labels_for_detected,
        colors=detected_colors,
        markerfacecolor=detected_colors,
        edgecolor=edgecolor,
    )

    # plot reduced missed spikes
    markerfacecolor = np.tile(np.array([[1], [1], [1]]), len(missed_colors)).T
    visualize_components(
        axis,
        component1=embedding_for_missed[:, 0],
        component2=embedding_for_missed[:, 1],
        labels=cell_labels_for_missed,
        colors=missed_colors,
        markerfacecolor=markerfacecolor,
        edgecolor=missed_colors,
    )

    # plot fit templates
    templates_embedding = embedding[n_spikes:, :]
    axis.plot(
        templates_embedding[:, 0],
        templates_embedding[:, 1],
        ".r",
        zorder=100,
        markersize=4,
        label="univ. templates",
    )
    plt.legend()
    return embedding


def pca_spikes_and_template_fits(
    axis,
    spike_instances,
    template_fits,
    labels,
    detection_status,
    detected_colors,
    missed_colors,
):
    """_summary_
    note: there is one template fit per spike

    Args:
        axis (_type_): _description_
        spike_instances (_type_): _description_
        template_fits (_type_): _description_
        labels (_type_): _description_
        detection_status (_type_): _description_

    Returns:
        _type_: _description_
    """
    # instantiate PCA
    pca_model = PCA(n_components=2)

    # create all instances
    instances = np.vstack([spike_instances, template_fits])

    # label spikes and fitted template
    n_spikes = int(len(instances) / 2)
    is_spike = np.hstack([np.ones(n_spikes), np.zeros(n_spikes)])

    # performs PCA
    pca_model.fit(instances)

    # take first two principal components
    PCA(n_components=2)

    # perform PCA
    embedding = pca_model.transform(instances)

    # separate detected and missed cell spike instances
    embed_for_detected = embedding[
        (detection_status == 1) & (is_spike == 1), :
    ]
    cell_labels_for_detected = labels[
        (detection_status == 1) & (is_spike == 1)
    ]

    embed_for_missed = embedding[(detection_status == 0) & (is_spike == 1), :]
    cell_labels_for_missed = labels[(detection_status == 0) & (is_spike == 1)]

    # isolate fitted template instances
    embed_for_template_fits = embedding[is_spike == 0, :]

    # plot reduced detected spikes
    edgecolor = np.tile(np.array([[1], [1], [1]]), len(detected_colors)).T
    visualize_components(
        axis,
        component1=embed_for_detected[:, 0],
        component2=embed_for_detected[:, 1],
        labels=cell_labels_for_detected,
        colors=detected_colors,
        markerfacecolor=detected_colors,
        edgecolor=edgecolor,
    )

    # plot reduced missed spikes
    markerfacecolor = np.tile(np.array([[1], [1], [1]]), len(missed_colors)).T
    visualize_components(
        axis,
        component1=embed_for_missed[:, 0],
        component2=embed_for_missed[:, 1],
        labels=cell_labels_for_missed,
        colors=missed_colors,
        markerfacecolor=markerfacecolor,
        edgecolor=missed_colors,
    )

    # plot fit templates
    axis.plot(
        embed_for_template_fits[:, 0],
        embed_for_template_fits[:, 1],
        ".r",
        zorder=100,
        markersize=4,
    )

    plt.legend()
    return embedding, pca_model


def tsne_spikes_and_template_fits(
    axis,
    spike_instances,
    template_fits,
    labels,
    detection_status,
    detected_colors,
    missed_colors,
):
    # create all instances
    instances = np.vstack([spike_instances, template_fits])

    # label spikes and fitted template
    n_spikes = int(len(instances) / 2)
    is_spike = np.hstack([np.ones(n_spikes), np.zeros(n_spikes)])

    # instantiate tsne
    tsne_model = TSNE(n_components=2, perplexity=70, random_state=2020)

    # perform t-SNE
    embedding = tsne_model.fit_transform(instances)

    # separate detected and missed cell spike instances
    embed_for_detected = embedding[
        (detection_status == 1) & (is_spike == 1), :
    ]
    cell_labels_for_detected = labels[
        (detection_status == 1) & (is_spike == 1)
    ]

    embed_for_missed = embedding[(detection_status == 0) & (is_spike == 1), :]
    cell_labels_for_missed = labels[(detection_status == 0) & (is_spike == 1)]

    # isolate fitted template instances
    embed_for_template_fits = embedding[is_spike == 0, :]

    # plot reduced detected spikes
    edgecolor = np.tile(np.array([[1], [1], [1]]), len(detected_colors)).T
    visualize_components(
        axis,
        component1=embed_for_detected[:, 0],
        component2=embed_for_detected[:, 1],
        labels=cell_labels_for_detected,
        colors=detected_colors,
        markerfacecolor=detected_colors,
        edgecolor=edgecolor,
    )

    # plot reduced missed spikes
    markerfacecolor = np.tile(np.array([[1], [1], [1]]), len(missed_colors)).T
    visualize_components(
        axis,
        component1=embed_for_missed[:, 0],
        component2=embed_for_missed[:, 1],
        labels=cell_labels_for_missed,
        colors=missed_colors,
        markerfacecolor=markerfacecolor,
        edgecolor=missed_colors,
    )

    # plot fit templates
    axis.plot(
        embed_for_template_fits[:, 0], embed_for_template_fits[:, 1], ".r"
    )

    plt.legend()
    return embedding, tsne_model


def visualize_components(
    ax, component1, component2, labels, colors, markerfacecolor, edgecolor
):
    """
    Plots a 2D representation of the data for visualization with categories
    labelled as different colors.

    Args:
      component1 (numpy array of floats) : Vector of component 1 scores
      component2 (numpy array of floats) : Vector of component 2 scores
      labels (numpy array of floats)     : Vector corresponding to categories of
                                           samples

    Returns:
      Nothing.

    """
    # get set of cell labels
    label_set = np.unique(labels)

    for ix, label_i in enumerate(label_set):
        # locate this cell label
        label_loc = labels == label_i

        # get coordinates
        x = component1[label_loc.squeeze()]
        y = component2[label_loc.squeeze()]

        # plot
        ax.plot(
            x,
            y,
            marker="o",
            color=colors[ix, :],
            markerfacecolor=markerfacecolor[ix, :],
            linestyle="none",
            label=label_i,
            markeredgecolor=edgecolor[ix, :],
        )

    # legend
    ax.set_xlabel("component1", fontsize=9)
    ax.set_ylabel("component2", fontsize=9)
