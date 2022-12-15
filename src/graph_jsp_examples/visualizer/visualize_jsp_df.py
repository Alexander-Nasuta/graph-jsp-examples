import pandas as pd

from graph_jsp_env.disjunctive_graph_jsp_visualizer import DisjunctiveGraphJspVisualizer

plotly_gantt_chart_data_dict = {
    'Task': {
        0: 'Job 0',
        1: 'Job 0',
        2: 'Job 0',
        3: 'Job 0',
        4: 'Job 1',
        5: 'Job 1',
        6: 'Job 1',
        7: 'Job 1'
    },
    'Start': {
        0: 5,
        1: 16,
        2: 21,
        3: 24,
        4: 0,
        5: 5,
        6: 21,
        7: 36
    },
    'Finish': {
        0: 16,
        1: 19,
        2: 24,
        3: 36,
        4: 5,
        5: 21,
        6: 28,
        7: 40
    },
    'Resource': {
        0: 'Machine 0',
        1: 'Machine 1',
        2: 'Machine 2',
        3: 'Machine 3',
        4: 'Machine 0',
        5: 'Machine 2',
        6: 'Machine 1',
        7: 'Machine 3'
    }
}

plotly_gantt_chart_df = pd.DataFrame.from_dict(plotly_gantt_chart_data_dict)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    c_map = plt.cm.get_cmap("gist_rainbow")  # select the desired cmap
    arr = np.linspace(0, 1, 4)  # create a list with numbers from 0 to 1 with n items
    colors = {f"Machine {resource}": c_map(val)[1:] for resource, val in enumerate(arr)}

    visualizer = DisjunctiveGraphJspVisualizer()
    visualizer.gantt_chart_console(df=plotly_gantt_chart_df, colors=colors)
