import matplotlib.pyplot as plt


def plot_predictions(predictions, mode_probs, gt, filename):
    for mode_index in range(mode_probs.shape[0]):
        plt.scatter(
            predictions[mode_index, :, 0], predictions[mode_index, :, 1],
            color='red', alpha=mode_probs[mode_index]
        )
    plt.scatter(gt[:, 0], gt[:, 1], color='green')
    plt.savefig(filename)
    plt.close()
