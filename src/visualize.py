from bokeh.io import export_svgs
from bokeh.palettes import Category10
from bokeh.plotting import figure, output_file, save

def render_line(df, x, y, legend, save_to=None, plot=None):
    """render line plot.

        Uses Bokeh as a bakend to render a line plot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame gholding the data
    x : str
        name of the column holding the data for the x axis.
    y : str
        name of the column holding the data for the y axis.
    legend : str
        name of the column containing the labels.
    save_to : str, Optional
        path with the file name where the plot is to be saved.
    plot : Bokeh plot, Optional
        plot to be used for plotting. If not defined then one is created.

    Return
    ------
        Bokeh plot object.
    """
    p = plot if plot is not None else figure()

    classes = list(set(df[legend]))
    colors = Category10[max([3, len(classes)])]

    for cls, color in zip(classes, colors):
        df_ = df.loc[df[legend] == cls]
        p.line(x=x, y=y, source=df_, line_width=2, legend_label=cls, color=color)

    if save_to is not None:
        output_file(save_to + ".html")
        save(p)
        p.output_backend = "svg"
        export_svgs(p, filename=save_to + ".svg")
    return p
