from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties
from optbinning.binning.binning_statistics import MulticlassBinningTable, COLORS_RGB, BinningTable, \
    ContinuousBinningTable

asset_home = Path(__file__).parent / 'assets'
font_path = asset_home / 'msyhl.ttc'
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_optbinning(binning_table, *args, **argv):
    if isinstance(binning_table, BinningTable):
        plot_binning_table(binning_table, *args, **argv)
    if isinstance(binning_table, MulticlassBinningTable):
        plot_multiclass_binning_table(binning_table, *args, **argv)
    if isinstance(binning_table, ContinuousBinningTable):
        plot_continuous_binning_table(binning_table, *args, **argv)


def _bin_str_label_format(bin_str, max_length=27):
    _bin_str = []
    for bs in bin_str:
        label = str(bs)
        if len(label) > max_length:
            label = label[:max_length] + '...'
        _bin_str.append(label)

    return _bin_str


def plot_binning_table(binning_table, add_special=True, add_missing=True, style="bin",
                       show_bin_labels=False, savefig=None, figsize=None, metric='mean',
                       dpi=800, image_format='png'):
    """Plot the binning table.

    Visualize records count and mean values.

    Parameters
    ----------
    metric : str, optional (default="mean")
        Supported metrics are "mean" to show the Mean value of the target
        variable in each bin, "iv" to show the IV of each bin and "woe" to
        show the Weight of Evidence (WoE) of each bin.

        .. versionadded:: 0.19.0

    add_special : bool (default=True)
        Whether to add the special codes bin.

    add_missing : bool (default=True)
        Whether to add the special values bin.

    style: str, optional (default="bin")
        Plot style. style="bin" shows the standard binning plot. If
        style="actual", show the plot with the actual scale, i.e, actual
        bin widths.

    show_bin_labels : bool (default=False)
        Whether to show the bin label instead of the bin id on the x-axis.
        For long labels (length > 27), labels are truncated.

        .. versionadded:: 0.15.1

    savefig : str or None (default=None)
        Path to save the plot figure.

    figsize : tuple or None (default=None)
        Size of the plot.
    """
    # _check_is_built(binning_table)

    if not isinstance(add_special, bool):
        raise TypeError("add_special must be a boolean; got {}."
                        .format(add_special))

    if not isinstance(add_missing, bool):
        raise TypeError("add_missing must be a boolean; got {}."
                        .format(add_missing))

    if style not in ("bin", "actual"):
        raise ValueError('Invalid value for style. Allowed string '
                         'values are "bin" and "actual".')

    if not isinstance(show_bin_labels, bool):
        raise TypeError("show_bin_labels must be a boolean; got {}."
                        .format(show_bin_labels))

    if show_bin_labels and style == "actual":
        raise ValueError('show_bin_labels only supported when '
                         'style="actual".')

    if figsize is not None:
        if not isinstance(figsize, tuple):
            raise TypeError('figsize argument must be a tuple.')

    if metric not in ("mean", "iv", "woe"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "mean", "iv" and "woe".')

    if style == "actual":
        # Hide special and missing bin
        add_special = False
        add_missing = False

        if binning_table.dtype == "categorical":
            raise ValueError('If style="actual", dtype must be numerical.')

        elif binning_table.min_x is None or binning_table.max_x is None:
            raise ValueError('If style="actual", min_x and max_x must be '
                             'provided.')

    if metric == "mean":
        metric_values = binning_table._mean
        metric_label = "Mean"
    elif metric == "woe":
        metric_values = binning_table._woe_values
        metric_label = "WoE"
    elif metric == "iv":
        metric_values = binning_table._iv_values
        metric_label = "IV"

    fig, ax1 = plt.subplots(figsize=figsize)

    if style == "bin":
        n_bins = len(binning_table.n_records)
        n_metric = n_bins - 1 - binning_table._n_specials

        if len(binning_table.cat_others):
            n_metric -= 1

        _n_records = list(binning_table.n_records)

        if not add_special:
            n_bins -= binning_table._n_specials
            for _ in range(binning_table._n_specials):
                _n_records.pop(-2)

        if not add_missing:
            _n_records.pop(-1)
            n_bins -= 1

        p1 = ax1.bar(range(n_bins), _n_records, color="tab:blue")

        handles = [p1[0]]
        labels = ['Count']

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)

        ax2 = ax1.twinx()

        ax2.plot(range(n_metric), metric_values[:n_metric],
                 linestyle="solid", marker="o", color="black")

        # Positions special and missing bars
        pos_special = 0
        pos_missing = 0

        if add_special:
            pos_special = n_metric
            if add_missing:
                pos_missing = n_metric + binning_table._n_specials
        elif add_missing:
            pos_missing = n_metric

        # Add points for others (optional), special and missing bin
        if len(binning_table.cat_others):
            pos_others = n_metric
            pos_special += 1
            pos_missing += 1

            p1[pos_others].set_alpha(0.5)

            ax2.plot(pos_others, metric_values[pos_others], marker="o",
                     color="black")

        if add_special:
            for i in range(binning_table._n_specials):
                p1[pos_special + i].set_hatch("/")

            handle_special = mpatches.Patch(hatch="/", alpha=0.1)
            label_special = "Bin special"

            for s in range(binning_table._n_specials):
                ax2.plot(pos_special + s, metric_values[pos_special + s],
                         marker="o", color="black")

        if add_missing:
            p1[pos_missing].set_hatch("\\")
            handle_missing = mpatches.Patch(hatch="\\", alpha=0.1)
            label_missing = "Bin missing"

            ax2.plot(pos_missing, metric_values[pos_missing], marker="o",
                     color="black")

        if add_special and add_missing:
            handles.extend([handle_special, handle_missing])
            labels.extend([label_special, label_missing])
        elif add_special:
            handles.extend([handle_special])
            labels.extend([label_special])
        elif add_missing:
            handles.extend([handle_missing])
            labels.extend([label_missing])

        ax2.set_ylabel(metric_label, fontsize=13)
        ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

        if show_bin_labels:
            if binning_table.dtype == "categorical":
                bin_str = _bin_str_label_format(binning_table._bin_str)
            else:
                bin_str = binning_table._bin_str

            if not add_special:
                bin_str = bin_str[:-2] + [bin_str[-1]]

            if not add_missing:
                bin_str = bin_str[:-1]

            ax1.set_xlabel("Bin", fontsize=12)
            ax1.set_xticks(np.arange(len(bin_str)))
            ax1.set_xticklabels(bin_str, rotation=45, ha="right")

    elif style == "actual":
        _n_records = binning_table.n_records[:-(binning_table._n_specials + 1)]

        n_splits = len(binning_table.splits)

        y_pos = np.empty(n_splits + 2)
        y_pos[0] = binning_table.min_x
        y_pos[1:-1] = binning_table.splits
        y_pos[-1] = binning_table.max_x

        width = y_pos[1:] - y_pos[:-1]
        y_pos2 = y_pos[:-1]

        p1 = ax1.bar(y_pos2, _n_records, width, color="tab:blue",
                     align="edge")

        handles = [p1[0]]
        labels = ['Count']

        ax1.set_xlabel("x", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)
        ax1.tick_params(axis='x', labelrotation=45)

        ax2 = ax1.twinx()

        for i in range(n_splits + 1):
            ax2.plot([y_pos[i], y_pos[i + 1]], [metric_values[i]] * 2,
                     linestyle="solid", color="black")

        ax2.plot(width / 2 + y_pos2,
                 metric_values[:-(binning_table._n_specials + 1)],
                 linewidth=0.75, marker="o", color="black")

        for split in binning_table.splits:
            ax2.axvline(x=split, color="black", linestyle="--",
                        linewidth=0.9)

        ax2.set_ylabel(metric_label, fontsize=13)

    plt.title(binning_table.name, fontsize=14)

    if show_bin_labels:
        legend_high = max(map(len, bin_str)) / 70 + 0.2
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -legend_high), ncol=2, fontsize=12)
    else:
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)

    if savefig is None:
        plt.show()
    else:
        plt.savefig(
            savefig, format=image_format,
            # dpi=dpi
        )
        plt.close()


def plot_multiclass_binning_table(binning_table, add_special=True, add_missing=True, show_bin_labels=False,
                                  savefig=None, figsize=None,
                                  dpi=800, image_format='png'):
    """Plot the binning table.

    Visualize event count and event rate values for each class.

    Parameters
    ----------
    add_special : bool (default=True)
        Whether to add the special codes bin.

    add_missing : bool (default=True)
        Whether to add the special values bin.

    show_bin_labels : bool (default=False)
        Whether to show the bin label instead of the bin id on the x-axis.
        For long labels (length > 27), labels are truncated.

        .. versionadded:: 0.15.1

    savefig : str or None (default=None)
        Path to save the plot figure.

    figsize : tuple or None (default=None)
        Size of the plot.
    """
    # _check_is_built(binning_table)

    if not isinstance(add_special, bool):
        raise TypeError("add_special must be a boolean; got {}."
                        .format(add_special))

    if not isinstance(add_missing, bool):
        raise TypeError("add_missing must be a boolean; got {}."
                        .format(add_missing))

    if not isinstance(show_bin_labels, bool):
        raise TypeError("show_bin_labels must be a boolean; got {}."
                        .format(show_bin_labels))

    if figsize is not None:
        if not isinstance(figsize, tuple):
            raise TypeError('figsize argument must be a tuple.')

    n_bins = len(binning_table._n_records)
    n_metric = n_bins - 1 - binning_table._n_specials
    n_classes = len(binning_table.classes)

    fig, ax1 = plt.subplots(figsize=figsize)

    colors = COLORS_RGB[:n_classes]
    colors = [tuple(c / 255. for c in color) for color in colors]

    if not add_special:
        n_bins -= binning_table._n_specials

    if not add_missing:
        n_bins -= 1

    _n_event = []
    for i in range(n_classes):
        _n_event_c = list(binning_table.n_event[:, i])
        if not add_special:
            for _ in range(binning_table._n_specials):
                _n_event_c.pop(-2)
        if not add_missing:
            _n_event_c.pop(-1)
        _n_event.append(np.array(_n_event_c))

    _n_event = np.array(_n_event)

    p = []
    cum_size = np.zeros(n_bins)
    for i, cl in enumerate(binning_table.classes):
        p.append(ax1.bar(range(n_bins), _n_event[i],
                         color=colors[i], bottom=cum_size))
        cum_size += _n_event[i]

    handles = [_p[0] for _p in p]
    labels = list(binning_table.classes)

    ax1.set_xlabel("Bin ID", fontsize=12)
    ax1.set_ylabel("Bin count", fontsize=13)

    ax2 = ax1.twinx()

    metric_values = binning_table._event_rate
    metric_label = "Event rate"

    for i, cl in enumerate(binning_table.classes):
        ax2.plot(range(n_metric), metric_values[:n_metric, i],
                 linestyle="solid", marker="o", color="black",
                 markerfacecolor=colors[i], markeredgewidth=0.5)

    # Add points for special and missing bin
    if add_special:
        pos_special = n_metric
        if add_missing:
            pos_missing = n_metric + binning_table._n_specials
    elif add_missing:
        pos_missing = n_metric

    if add_special:
        for _p in p:
            for i in range(binning_table._n_specials):
                _p[pos_special + i].set_hatch("/")

        handle_special = mpatches.Patch(hatch="/", alpha=0.1)
        label_special = "Bin special"

        for i, cl in enumerate(binning_table.classes):
            for s in range(binning_table._n_specials):
                ax2.plot(pos_special + s, metric_values[pos_special + s, i],
                         marker="o", color=colors[i])

    if add_missing:
        for _p in p:
            _p[pos_missing].set_hatch("\\")

        handle_missing = mpatches.Patch(hatch="\\", alpha=0.1)
        label_missing = "Bin missing"

        for i, cl in enumerate(binning_table.classes):
            ax2.plot(pos_missing, metric_values[pos_missing, i],
                     marker="o", color=colors[i])

    if add_special and add_missing:
        handles.extend([handle_special, handle_missing])
        labels.extend([label_special, label_missing])
    elif add_special:
        handles.extend([handle_special])
        labels.extend([label_special])
    elif add_missing:
        handles.extend([handle_missing])
        labels.extend([label_missing])

    ax2.set_ylabel(metric_label, fontsize=13)
    ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

    if show_bin_labels:
        bin_str = binning_table._bin_str
        if not add_special:
            bin_str = bin_str[:-2] + [bin_str[-1]]

        if not add_missing:
            bin_str = bin_str[:-1]

        ax1.set_xlabel("Bin", fontsize=12)
        ax1.set_xticks(np.arange(len(bin_str)))
        ax1.set_xticklabels(bin_str, rotation=45, ha="right")

    plt.title(binning_table.name, fontsize=14)

    if show_bin_labels:
        legend_high = max(map(len, binning_table._bin_str)) / 70 + 0.2
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -legend_high), ncol=2, fontsize=12)
    else:
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)

    if savefig is None:
        plt.show()
    else:

        plt.savefig(
            savefig, format=image_format,
            # dpi=dpi
        )
        plt.close()


def plot_continuous_binning_table(binning_table, add_special=True, add_missing=True, style="bin",
                                  show_bin_labels=False, savefig=None, figsize=None, metric='mean',
                                  dpi=800, image_format='png'):
    """Plot the binning table.

    Visualize records count and mean values.

    Parameters
    ----------
    metric : str, optional (default="mean")
        Supported metrics are "mean" to show the Mean value of the target
        variable in each bin, "iv" to show the IV of each bin and "woe" to
        show the Weight of Evidence (WoE) of each bin.

        .. versionadded:: 0.19.0

    add_special : bool (default=True)
        Whether to add the special codes bin.

    add_missing : bool (default=True)
        Whether to add the special values bin.

    style: str, optional (default="bin")
        Plot style. style="bin" shows the standard binning plot. If
        style="actual", show the plot with the actual scale, i.e, actual
        bin widths.

    show_bin_labels : bool (default=False)
        Whether to show the bin label instead of the bin id on the x-axis.
        For long labels (length > 27), labels are truncated.

        .. versionadded:: 0.15.1

    savefig : str or None (default=None)
        Path to save the plot figure.

    figsize : tuple or None (default=None)
        Size of the plot.
    """
    # _check_is_built(binning_table)

    if not isinstance(add_special, bool):
        raise TypeError("add_special must be a boolean; got {}."
                        .format(add_special))

    if not isinstance(add_missing, bool):
        raise TypeError("add_missing must be a boolean; got {}."
                        .format(add_missing))

    if style not in ("bin", "actual"):
        raise ValueError('Invalid value for style. Allowed string '
                         'values are "bin" and "actual".')

    if not isinstance(show_bin_labels, bool):
        raise TypeError("show_bin_labels must be a boolean; got {}."
                        .format(show_bin_labels))

    if show_bin_labels and style == "actual":
        raise ValueError('show_bin_labels only supported when '
                         'style="actual".')

    if figsize is not None:
        if not isinstance(figsize, tuple):
            raise TypeError('figsize argument must be a tuple.')

    if metric not in ("mean", "iv", "woe"):
        raise ValueError('Invalid value for metric. Allowed string '
                         'values are "mean", "iv" and "woe".')

    if style == "actual":
        # Hide special and missing bin
        add_special = False
        add_missing = False

        if binning_table.dtype == "categorical":
            raise ValueError('If style="actual", dtype must be numerical.')

        elif binning_table.min_x is None or binning_table.max_x is None:
            raise ValueError('If style="actual", min_x and max_x must be '
                             'provided.')

    if metric == "mean":
        metric_values = binning_table._mean
        metric_label = "Mean"
    elif metric == "woe":
        metric_values = binning_table._woe_values
        metric_label = "WoE"
    elif metric == "iv":
        metric_values = binning_table._iv_values
        metric_label = "IV"

    fig, ax1 = plt.subplots(figsize=figsize)

    if style == "bin":
        n_bins = len(binning_table.n_records)
        n_metric = n_bins - 1 - binning_table._n_specials

        if len(binning_table.cat_others):
            n_metric -= 1

        _n_records = list(binning_table.n_records)

        if not add_special:
            n_bins -= binning_table._n_specials
            for _ in range(binning_table._n_specials):
                _n_records.pop(-2)

        if not add_missing:
            _n_records.pop(-1)
            n_bins -= 1

        p1 = ax1.bar(range(n_bins), _n_records, color="tab:blue")

        handles = [p1[0]]
        labels = ['Count']

        ax1.set_xlabel("Bin ID", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)

        ax2 = ax1.twinx()

        ax2.plot(range(n_metric), metric_values[:n_metric],
                 linestyle="solid", marker="o", color="black")

        # Positions special and missing bars
        pos_special = 0
        pos_missing = 0

        if add_special:
            pos_special = n_metric
            if add_missing:
                pos_missing = n_metric + binning_table._n_specials
        elif add_missing:
            pos_missing = n_metric

        # Add points for others (optional), special and missing bin
        if len(binning_table.cat_others):
            pos_others = n_metric
            pos_special += 1
            pos_missing += 1

            p1[pos_others].set_alpha(0.5)

            ax2.plot(pos_others, metric_values[pos_others], marker="o",
                     color="black")

        if add_special:
            for i in range(binning_table._n_specials):
                p1[pos_special + i].set_hatch("/")

            handle_special = mpatches.Patch(hatch="/", alpha=0.1)
            label_special = "Bin special"

            for s in range(binning_table._n_specials):
                ax2.plot(pos_special + s, metric_values[pos_special + s],
                         marker="o", color="black")

        if add_missing:
            p1[pos_missing].set_hatch("\\")
            handle_missing = mpatches.Patch(hatch="\\", alpha=0.1)
            label_missing = "Bin missing"

            ax2.plot(pos_missing, metric_values[pos_missing], marker="o",
                     color="black")

        if add_special and add_missing:
            handles.extend([handle_special, handle_missing])
            labels.extend([label_special, label_missing])
        elif add_special:
            handles.extend([handle_special])
            labels.extend([label_special])
        elif add_missing:
            handles.extend([handle_missing])
            labels.extend([label_missing])

        ax2.set_ylabel(metric_label, fontsize=13)
        ax2.xaxis.set_major_locator(mtick.MultipleLocator(1))

        if show_bin_labels:
            if binning_table.dtype == "categorical":
                bin_str = _bin_str_label_format(binning_table._bin_str)
            else:
                bin_str = binning_table._bin_str

            if not add_special:
                bin_str = bin_str[:-2] + [bin_str[-1]]

            if not add_missing:
                bin_str = bin_str[:-1]

            ax1.set_xlabel("Bin", fontsize=12)
            ax1.set_xticks(np.arange(len(bin_str)))
            ax1.set_xticklabels(bin_str, rotation=45, ha="right")

    elif style == "actual":
        _n_records = binning_table.n_records[:-(binning_table._n_specials + 1)]

        n_splits = len(binning_table.splits)

        y_pos = np.empty(n_splits + 2)
        y_pos[0] = binning_table.min_x
        y_pos[1:-1] = binning_table.splits
        y_pos[-1] = binning_table.max_x

        width = y_pos[1:] - y_pos[:-1]
        y_pos2 = y_pos[:-1]

        p1 = ax1.bar(y_pos2, _n_records, width, color="tab:blue",
                     align="edge")

        handles = [p1[0]]
        labels = ['Count']

        ax1.set_xlabel("x", fontsize=12)
        ax1.set_ylabel("Bin count", fontsize=13)
        ax1.tick_params(axis='x', labelrotation=45)

        ax2 = ax1.twinx()

        for i in range(n_splits + 1):
            ax2.plot([y_pos[i], y_pos[i + 1]], [metric_values[i]] * 2,
                     linestyle="solid", color="black")

        ax2.plot(width / 2 + y_pos2,
                 metric_values[:-(binning_table._n_specials + 1)],
                 linewidth=0.75, marker="o", color="black")

        for split in binning_table.splits:
            ax2.axvline(x=split, color="black", linestyle="--",
                        linewidth=0.9)

        ax2.set_ylabel(metric_label, fontsize=13)

    plt.title(binning_table.name, fontsize=14)

    if show_bin_labels:
        legend_high = max(map(len, bin_str)) / 70 + 0.2
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -legend_high), ncol=2, fontsize=12)
    else:
        plt.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)

    if savefig is None:
        plt.show()
    else:

        plt.savefig(
            savefig, format=image_format,
            # dpi=dpi
        )
        plt.close()
