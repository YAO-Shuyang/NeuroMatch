import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np


def _clear_spines(ax: Axes, close_spines: list[str] = ['top', 'right', 'bottom']) -> Axes:
    for i in close_spines:
        ax.spines[i].set_visible(False)
    return ax

def LocTimeCurve(data: dict, axes: np.ndarray, plot_range: np.ndarray, cell_indexes: np.ndarray,
                 line_kwargs = {'markeredgewidth': 0, 'markersize': 1, 'color': 'black'},
                 bar_kwargs = {'markeredgewidth': 3, 'markersize': 5}) -> np.ndarray:
    assert plot_range.shape[0] == cell_indexes.shape[0] and axes.shape[0] == cell_indexes.shape[0]
    maze_type = data['maze_type'][0]
    
    for i, j in enumerate(plot_range):
        axes[6-i].clear()
        if cell_indexes[i] == 0:
            axes[6-i] = _clear_spines(axes[6-i], close_spines=['top', 'right', 'bottom', 'left'])
            axes[6-i].set_xticks([])
            axes[6-i].set_yticks([])
            continue
        
        axes[6-i] = _clear_spines(axes[6-i])
        Spikes = data['Spikes'][j]
        linearized_x = data['linearized_x'][j]
        ms_time = data['ms_time_behav'][j]
        axes[6-i].plot(linearized_x, ms_time/1000, 'o', **line_kwargs)
        
        idx = np.where(Spikes[int(cell_indexes[i])-1, :] == 1)[0]
        axes[6-i].plot(linearized_x[idx], ms_time[idx]/1000, '|', color='red', **bar_kwargs)
        axes[6-i].set_ylabel(f"Cell {int(cell_indexes[i])}")
        t_max = int(np.nanmax(ms_time)/1000)
        axes[6-i].set_xticks([])
        axes[6-i].set_yticks([])
        axes[6-i].set_ylim([0, t_max])
        axes[6-i].set_xlim([0.5, 111.5]) if maze_type == 1 else axes[6-i].set_xlim([0.5, 101.5])
        
    return axes