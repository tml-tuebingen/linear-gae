import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
# noinspection PyPackageRequirements
import scipy

from sklearn.manifold import SpectralEmbedding

#mpl.use('agg')

COLORS = mpl.cm.get_cmap('Set1')


# noinspection PyArgumentList
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=int)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def plot_state(acc, loss, emb, original, clusters):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    flat_ax = ax.flatten()

    flat_ax[0].plot(loss, label='loss')
    flat_ax[0].set_yscale('log')
    flat_ax[0].legend()

    flat_ax[1].plot(acc, label='acc')
    flat_ax[1].set_ylim([0, 1])
    flat_ax[1].legend()

    flat_ax[2].set_title("recovered latent space")
    flat_ax[3].set_title("original embedding")

    for c in np.unique(clusters):
        flat_ax[2].scatter(emb[clusters == c, 0], emb[clusters == c, 1], label=c, color=COLORS(c))
        flat_ax[3].scatter(original[clusters == c, 0], original[clusters == c, 1], label=c, color=COLORS(c))

    flat_ax[2].legend()
    flat_ax[3].legend()

    flat_ax[2].legend(loc=1)
    flat_ax[3].legend(loc=1)

    return fig


def plot_results(emb, acc, test_roc, test_ap, adj, clusters):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('final train acc: %.4f test ap: %.4f, test roc: %.4f' % (acc, test_ap, test_roc))
    flat_ax = ax.flatten()

    for c in np.unique(clusters):
        # evaluate if distribution is standard normal
        # noinspection PyUnresolvedReferences
        print("class %s is normally distributed %s" % (
            int(c),
            [bool(scipy.stats.normaltest(emb[clusters == c, i].reshape(-1, 1), axis=0)[1] < 1e-3) for i in [0, 1]]))
        flat_ax[0].set_title("recovered latent space")
        flat_ax[0].scatter(emb[clusters == c, 0], emb[clusters == c, 1], label=c, color=COLORS(c))
    try:
        min_val = np.min(emb)
        max_val = np.max(emb)
        diff = max_val - min_val
        flat_ax[0].set_ylim([min_val - 0.1 * diff, max_val + 0.1 * diff])
        flat_ax[0].set_xlim([min_val - 0.1 * diff, max_val + 0.1 * diff])
        flat_ax[0].legend()
    except UserWarning:
        pass

    # original embedding if existing
    embedding = SpectralEmbedding(n_components=2, affinity='precomputed')
    embedding = embedding.fit_transform(adj)
    for c in np.unique(clusters):
        flat_ax[1].set_title("spectral embedding")
        flat_ax[1].scatter(embedding[clusters == c, 0], embedding[clusters == c, 1], label=c,
                           color=COLORS(c))
        flat_ax[1].legend()

    fig.tight_layout(pad=3)

    return fig


def save_gif(filenames, experiment_name):
    # save trained embedding as animated visualization
    with imageio.get_writer(experiment_name + '.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


def plot_state_small(emb, clusters, epoch, adj=None):
    if adj is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

    fig.suptitle(epoch)
    ax1.set_title("recovered latent space")

    for c in np.unique(clusters):
        if adj is not None:
            deg = np.array(np.sum(adj, axis=1)[clusters == c] - 1).flatten()

            im = ax1.scatter(emb[clusters == c, 0], emb[clusters == c, 1], label=c,
                             c=deg)
            # noinspection PyUnboundLocalVariable
            ax2.set_title("degree vs vector norm")
            ax2.scatter(np.linalg.norm(emb, axis=1)[clusters == c], deg, label=c,
                        c=deg)
            ax2.set_ylabel("degree")
            ax2.set_xlabel("vector norm")

        else:
            ax1.scatter(emb[clusters == c, 0], emb[clusters == c, 1], label=c, color=COLORS(c))

        min_val = np.min(emb)
        max_val = np.max(emb)
        diff = max_val - min_val
        ax1.set_ylim([min_val - 0.1 * diff, max_val + 0.1 * diff])
        ax1.set_xlim([min_val - 0.1 * diff, max_val + 0.1 * diff])

    if adj is not None:
        # noinspection PyUnboundLocalVariable
        cax = fig.colorbar(im, ax=ax2)
        cax.set_ticks(np.arange(0, max(adj.sum(axis=1)) + 1, 5))
        cax.set_label('node degree')

    ax1.legend()
    ax1.legend(loc=1)

    return fig


def plot_state_small_colored(emb, colors):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.set_title("recovered latent space")

        plt.scatter(emb[:, 0], emb[:, 1], c=colors)
        min_val = np.min(emb)
        max_val = np.max(emb)
        diff = max_val - min_val
        ax.set_ylim([min_val - 0.1 * diff, max_val + 0.1 * diff])
        ax.set_xlim([min_val - 0.1 * diff, max_val + 0.1 * diff])

        plt.clim(-0.5, 0.5)
        plt.colorbar()
    except TypeError:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.set_title("recovered latent space")

    return fig