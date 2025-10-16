import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from matplotlib.colors import Colormap

from typing import List, Tuple, Dict, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


FAULT_LABEL = '#Faults'
X_LABEL = 'Iterations'
AXIS_LABEL_FONTSIZE = 18
TITLE_LABEL_FONTSIZE = 30


MDPFUZZ_COLOR = 'deep sky blue'
FUZZER_COLOR = 'fire engine red'
RT_COLOR = 'shamrock green'


def get_method_colors():
    '''Returns the colors for Fuzzer, MDPFuzz and RT (alphabetical order).'''
    return [f'xkcd:{c}' for c in [FUZZER_COLOR, MDPFUZZ_COLOR, RT_COLOR]]


def get_tau_colors():
    colors = ['reddish', 'dark sky blue', 'emerald']
    return [f'xkcd:{c}' for c in colors]


def get_gamma_colors():
    colors = ['pastel green', 'goldenrod', 'blood orange', 'ruby']
    return [f'xkcd:{c}' for c in colors]


def get_colors(n: int = 2, cmap: Union[str, Colormap] = plt.cm.jet):
    '''Return a list of @n RGBA colors by sampling evenly in @cmap (default to "jet").'''
    if isinstance(cmap, str):
        cmap: Colormap = plt.cm.get_cmap(cmap)
    return [cmap(i) for i in np.linspace(0, 1, n)]


def scientific_notation(x, pos):
    if x == 0:
        return '0'
    elif x < 1e3:
        return '{:.0f}'.format(x)
    else:
        exp = int(np.log10(x))
        coeff = x / 10**exp
        return r'${:.0f} \times 10^{{{}}}$'.format(coeff, exp)


def as_thousands_notation(x, pos):
    if x < 1000:
        return '{:.0f}'.format(x)
    else:
        return '{:.0f}K'.format(x / 1000)


##### RQ2 ######

def plot_results(
        use_cases: List[str],
        results: List[Tuple],
        color: Tuple,
        label: str,
        vertical: bool = False
    ):
    '''
    Plots the results of a method for each use-case.
    A chart per use-case; presented horizontally.
    '''
    n = len(use_cases)
    if vertical:
        fig, axs = plt.subplots(nrows=n, figsize=(6, 5 * n), sharex=True)
    else:
        fig, axs = plt.subplots(ncols=n, figsize=(5 * n, 6), sharex=True)

    [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs.flat]

    if vertical:
        axs[-1].set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONTSIZE + 5)
    else:
        axs[0].set_ylabel(FAULT_LABEL, fontsize=AXIS_LABEL_FONTSIZE + 5)

    # iterates over the use-cases
    for u in range(n):
        data = results[u]
        # labeling
        axs[u].set_title(use_cases[u], fontsize=TITLE_LABEL_FONTSIZE)#, loc='left')
        if vertical:
            axs[u].set_ylabel(FAULT_LABEL, fontsize=AXIS_LABEL_FONTSIZE + 5)
        else:
            axs[u].set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONTSIZE + 5)
        y, perc_25, perc_75 = data
        # over iterations
        x = np.arange(0, len(y))
        axs[u].plot(x, y, color=color, label=label)
        axs[u].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
        # axs[u].set_xticks(ticks)
        # axs[u].set_xticklabels(ticks)

    for ax in axs:
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=17)

    return (fig, axs)


def plot_results_as_grid(
        use_cases: List[str],
        results: List[Tuple],
        color: Tuple,
        label: str
    ):
    '''
    Plots the results of a method for each use-case.
    A chart per use-case; presented in the grid.
    '''
    n = len(use_cases)
    fig = plt.figure(figsize=(30, 10))
    gs = GridSpec(2, (2 * n), figure=fig)

    index = int(n / 2)
    axs = []
    size = 2
    for i in range(0, index + 2 * size + 1, size):
        axs.append(fig.add_subplot(gs[0, i:(i+size)]))
    for i in range(1, index + 2 * size, size):
        axs.append(fig.add_subplot(gs[1, i:(i+size)]))

    [ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1) for ax in axs]


    axs[0].set_ylabel(FAULT_LABEL, fontsize=AXIS_LABEL_FONTSIZE + 4)
    axs[1 + index].set_ylabel(FAULT_LABEL, fontsize=AXIS_LABEL_FONTSIZE + 4)

    # iterates over the use-cases
    for u in range(n):
        data = results[u]
        # labeling
        axs[u].set_title(use_cases[u], fontsize=TITLE_LABEL_FONTSIZE - 5)#, loc='left')
        axs[u].set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONTSIZE + 4)
        y, perc_25, perc_75 = data
        # over iterations
        x = np.arange(0, len(y))
        axs[u].plot(x, y, color=color, label=label)
        axs[u].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
        axs[u].tick_params(axis='both', labelsize=18)
    fig.tight_layout()
    return (fig, axs)



def adds_results_to_axs(
        axs: np.ndarray,
        results: List[Tuple],
        color: Tuple,
        label: str
    ):
    '''
    Adds results to axes.
    It assumes the axes of shape (1, num_use_cases) and that results are sorted w.r.t axes.
    '''
    ncols = len(axs)
    assert len(results) == ncols, len(results)
    for r in range(ncols):
        data = results[r]
        y, perc_25, perc_75 = data
        x = np.arange(0, len(y))
        axs[r].plot(x, y, color=color, label=label)
        axs[r].fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)
    return axs


def create_bar_plot(data: List[Dict],
                    use_case_names: List[str],
                    method_colors: Dict[str, Tuple[float, float, float, float]]) -> Tuple[plt.Figure, plt.Axes]:
    '''
    Generates a bar plot with statistical results for each method across different use-cases.

    Parameters:
    -----------
    data : List[List[Dict[str, Union[float, Tuple[float, float, float]]]]]
        A list of lists of dictionaries containing statistical results for each method across use-cases.

    use_case_names : List[str]
        Names of the use-cases.

    method_colors : Dict[str, Tuple[float, float, float, float]]
        Colors for each method represented as RGBA tuples.

    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        A tuple containing the Matplotlib figure and its axes.
    '''
    num_use_cases = len(use_case_names)
    method_names = list(method_colors.keys())
    num_methods = len(method_names)
    bar_width = 0.2
    x = np.arange(num_use_cases)

    fig, ax = plt.subplots(figsize=(num_use_cases * 1.3, 6))

    for i in range(num_methods):
        method_name = method_names[i]
        method_results = data[i]
        color = method_colors[method_name]
        offset = (i - num_methods / 2) * bar_width + bar_width / 2
        bar_positions = x + offset
        # run with low alpha
        heights = []
        errors = []
        for use_case in use_case_names:
            m, err = method_results[use_case]['run']
            m /= 60
            err /= 60
            heights.append(m)
            errors.append(err)
        ax.bar(bar_positions, heights, bar_width, yerr=errors, color=color, label=method_name)

        heights = []
        errors = []
        for use_case in use_case_names:
            m, err = method_results[use_case]['test']
            m /= 60
            err /= 60
            heights.append(m)
            errors.append(err)
        ax.bar(bar_positions, heights, bar_width, yerr=errors, color=color, edgecolor='black', hatch='///')

        heights2 = []
        errors = []
        for use_case in use_case_names:
            m, err = method_results[use_case]['cov']
            m /= 60
            err /= 60
            heights2.append(m)
            errors.append(err)
        ax.bar(bar_positions, heights2, bar_width, yerr=errors, bottom=heights, color=color, edgecolor='black', hatch='\\\\\\')


    # sets the labels and title
    # ax.set_xlabel('Use Cases')
    ax.set_ylabel('Time (min.)')
    # ax.set_title('Run times for Each Method')
    ax.set_xticks(x)
    ax.set_xticklabels(use_case_names)
    # sets the legend
    cov_patch = mpatches.Patch(facecolor=(0,0,0,0), edgecolor=(0,0,0,1), hatch='///', label='Coverage time')
    test_patch = mpatches.Patch(facecolor=(0,0,0,0), edgecolor=(0,0,0,1), hatch='\\\\\\', label='Test time')
    handles = ax.get_legend_handles_labels()[0]
    all_handles = handles + [cov_patch, test_patch]
    ax.legend(handles=all_handles)
    return fig, ax


def create_bar_plots(
        data: Dict[str, List[Tuple]],
        method_names: List[str],
        time_colors: Dict[str, Tuple[float, float, float, float]]
    ) -> Tuple[plt.Figure, plt.Axes]:
    '''
    Generates bar plots with statistical results for each use-case.
    '''
    num_use_cases = len(data)
    num_methods = len(method_names)
    bar_width = 0.65
    x = np.arange(num_methods)

    fig, axs = plt.subplots(ncols=num_use_cases, figsize=(num_use_cases * 3.8, 4))
    axs[0].set_ylabel('Time (min)', fontsize=AXIS_LABEL_FONTSIZE)

    for i, (use_case, times_of_methods) in enumerate(data.items()):
        ax = axs[i]
        ax.set_title(use_case, fontsize=TITLE_LABEL_FONTSIZE - 4)
        for j, (label, color) in enumerate(time_colors.items()):
            times = [t[j] for t in times_of_methods]

            # the last times are on top of the previous bars
            if j == len(time_colors) - 1:
                last_heights = heights.copy()

            heights = [v[0] for v in times]
            yerrors = [v[1] for v in times]

            if j == len(time_colors) - 1:
                ax.bar(x, heights, bar_width, bottom=last_heights, color=color)
            else:
                ax.bar(x, heights, bar_width, color=color)
        # x ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=25, fontsize=AXIS_LABEL_FONTSIZE)
        # background
        ax.set_facecolor('lightgrey')
        ax.grid(axis='y', color='white')
        ax.set_axisbelow(True)
        # hides the frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # increases font size of the y axis
        ax.tick_params(axis='y', labelsize=12)
        y_tick_labels = ax.get_yticklabels()
        if len(y_tick_labels) > 7:
            ax.locator_params(axis='y', nbins=5)

    legend_elements = [
        Line2D([0], [0], marker='s', color='white', label=label, markerfacecolor=color, markersize=20)
        for (label, color) in time_colors.items()
    ]

    legend_fontsize = 15
    legend = axs[-1].legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        prop={'size': legend_fontsize},
        title='Time',
        shadow=False,
        labelspacing=1.0,
        handletextpad=0.0001
    )
    legend.get_title().set_fontsize(str(legend_fontsize))
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('1')
    legend_frame.set_edgecolor('1')

    fig.subplots_adjust(wspace=0.001)
    return fig, ax



##### RQ3 #####


def plot_k_g_analysis(
        use_cases: List[str],
        results: List[Dict],
        colors_dict: Dict[str, Tuple[float]],
        y_axis: Literal['k', 'gamma'],
        use_case_labels: List[str] = None,
        **kwargs):

    k_keys: List[str] = np.unique([d['k'] for d in results]).tolist()
    gamma_keys: List[str] = np.unique([d['gamma'] for d in results]).tolist()

    if use_case_labels is None:
        use_case_labels = use_cases
    else:
        assert len(use_case_labels) == len(use_cases)

    x_labels = use_case_labels.copy()
    x_labels.sort()
    num_use_cases = len(use_cases)
    parameter_values = k_keys if y_axis == 'k' else gamma_keys
    label_key = 'k' if y_axis != 'k' else 'gamma'
    labels = k_keys if y_axis != 'k' else gamma_keys

    nrows = len(parameter_values)
    ncols = num_use_cases
    if num_use_cases == 1:
        fig, row_axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*num_use_cases, 5*len(parameter_values)))
        axs = np.array([[ax] for ax in row_axs])
    else:
        # fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*num_use_cases, 4*len(parameter_values)), sharex=True)#, sharey='col')
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*num_use_cases, 5*len(parameter_values)), sharex=True, sharey='col')

    for ax in axs.flat:
        ax.grid(axis='y', color='0.9', linestyle='-', linewidth=1)

    # fig.supylabel(y_axis.capitalize(), fontsize=AXIS_LABEL_FONTSIZE)
    # fig.supxlabel('Use-cases', fontsize=AXIS_LABEL_FONTSIZE)

    # iterates column-wise
    for i, p in enumerate(parameter_values):
        if y_axis == 'k':
            y_label = rf'$K={p}$'
        else:
            y_label = rf'$\gamma={p}$'
        axs[i][0].set_ylabel(y_label, fontsize=TITLE_LABEL_FONTSIZE - 2)
        for u in range(num_use_cases):
            use_case = use_case_labels[u]
            ax = axs[i][u]
            axs[-1][u].set_xlabel(use_case, fontsize=TITLE_LABEL_FONTSIZE)
            for p2 in labels:
                result_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [d[use_cases[u]] for d in results if d[label_key] == p2 and d[y_axis] == p]
                if len(result_list) == 0:
                    print(f'No result found for configuration {y_axis}: {p}, {label_key}: {p2} on {use_case} ({use_cases[u]})')
                    continue
                assert len(result_list) == 1, f'Multiple results for configuration {y_axis}: {p}, {label_key}: {p2} on {use_case} ({len(result_list)})'
                #TODO: dealing with color for each parameter value
                color = colors_dict[p2]
                label = rf'$\gamma={p2}$' if y_axis == 'k' else rf'$K={p2}$'
                y, perc_25, perc_75 = result_list[0]
                x = np.arange(len(y))
                ax.plot(x, y, color=color, label=label, linewidth=2.5)
                ax.fill_between(x, perc_25, perc_75, alpha=0.25, linewidth=0, color=color)

    # ax = axs.flat[np.argmax([len(ax.get_lines()) for ax in axs.flat])]
    flat_axs = axs.flat
    for i in range(0, len(flat_axs), 2):
        ax = flat_axs[i]
        legend = ax.legend(prop={'size': 16}, loc="upper center")
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('0.9')
        legend_frame.set_edgecolor('0.9')
        # change the line width for the legend
        for line in legend.get_lines():
            line.set_linewidth(6.0)

    y_tick_padding = -30

    for ax in flat_axs:
        ax.tick_params(axis='both', labelsize=16)

        # moves the latter and their labels inside the plot
        ax.tick_params(axis="y", direction="in")
        for tick in ax.yaxis.get_major_ticks():
            tick.set_pad(y_tick_padding)
            tick.label1.set_horizontalalignment("center")

    for ax in axs[0]:
        init_ticks = ax.get_yticks()
        labels = init_ticks[1:-1]
        ticks = [int(t) for t in labels]

        ax.set_yticks(ticks)
        ax.set_yticklabels([""] + [str(t) for t in ticks[1:]])

    title = kwargs.get('title', None)
    if title is not None:
        fig.suptitle(title, fontsize=TITLE_LABEL_FONTSIZE)

    fig.tight_layout()

    filename = kwargs.get('filename', None)
    if filename is not None:
        filename = f'{filename}.png' if not filename.endswith('.png') else filename
        fig.savefig(filename)

    return (fig, axs)