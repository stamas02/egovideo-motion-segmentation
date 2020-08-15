
import matplotlib.pyplot as plt

def plot(df, x, y, color = None, save_to = None):
    ax = plt.gca()


    if color is None:
        df.plot(kind='line', x=x, y=y, ax=ax)
    else:
        legend_labels = set(df[color])
        for l in legend_labels:
            df_ = df.loc[df[color] == l]
            df_.plot(kind='line', x=x, y=y, ax=ax)

    if save_to is not None:
        plt.savefig(save_to,
                    bbox_inches='tight', pad_inches=0.02
                    )
    plt.close()