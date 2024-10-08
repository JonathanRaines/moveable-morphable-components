from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.express.colors import qualitative, sample_colorscale

from moveable_morphable_components import components

COLOURS = qualitative.Pastel[:2]
TRANSPARENT = "rgba(0,0,0,0)"


# def component_image(component_list, coords, dimensions) -> go.Figure:
#     fig = go.Figure()

#     tdf_values = evaluate_topology_description_functions(component_list, coords)
#     contour_settings = {"start": 0, "end": 1, "size": 2}
#     colour_scales = [[[0, TRANSPARENT], [1, c]] for c in COLOURS]

#     traces = [
#         go.Contour(
#             z=sdf.T,
#             colorscale=colour_scales[i % len(colour_scales)],
#             contours=contour_settings,
#             line_smoothing=0,
#             showscale=False,
#             showlegend=False,
#             x=np.linspace(0, dimensions[0], coords[0].shape[0]),
#             y=np.linspace(0, dimensions[1], coords[0].shape[1]),
#         )
#         for i, sdf in enumerate(tdf_values)
#     ]

#     fig.add_traces(traces)
#     fig.update_layout(
#         dict(
#             template="simple_white",
#             plot_bgcolor=TRANSPARENT,
#             paper_bgcolor=TRANSPARENT,
#         ),
#     )

#     return fig


def save_component_animation(
    design_var_history,
    component_groups: list[components.ComponentGroup],
    coords,
    dimensions,
    duration: int = 5_000,
    filename: str = "mmc",
) -> None:
    num_components: int = sum(
        [cg.num_design_variables for cg in component_groups])
    frames: list[Image.Image] = []
    contour_settings = dict(start=0, end=1, size=2)
    colour_scales = [
        [[0, TRANSPARENT], [1, c]]
        for c in sample_colorscale(qualitative.Pastel, num_components)
    ]

    x_step = coords[0][1]
    n_x = int(dimensions[0] / x_step) + 1
    y_step = coords[1][n_x]
    n_y = int(dimensions[1] / y_step) + 1

    for global_design_vars in design_var_history:
        fig = go.Figure()

        group_design_var_count = [
            cg.num_design_variables for cg in component_groups]
        design_vars_per_component = [
            len(cg.free_variable_col_indexes) for cg in component_groups
        ]
        grouped_design_vars = np.split(
            global_design_vars, np.cumsum(group_design_var_count), axis=0,
        )
        grouped_design_vars = [
            gdv.reshape(-1, n)
            for gdv, n in zip(grouped_design_vars, design_vars_per_component)
        ]
        sdfs = []
        for i, cg in enumerate(component_groups):
            sdfs.extend([cg.tdf(design_vars).reshape(n_x, n_y, order="F")
                        for design_vars in grouped_design_vars[i]])

        traces = [
            go.Contour(
                z=sdf.T,
                colorscale=colour_scales[i % len(colour_scales)],
                contours=contour_settings,
                line_smoothing=0,
                showscale=False,
                showlegend=False,
                x=np.linspace(0, dimensions[0], n_x),
                y=np.linspace(0, dimensions[1], n_y),
            )
            for i, sdf in enumerate(sdfs)
        ]

        fig.add_traces(traces)
        fig.update_layout(
            {
                "template": "simple_white",
                # plot_bgcolor=TRANSPARENT, #TODO(JonathanRaines): PIL seems to layer images so can't have transparent background
                # paper_bgcolor=TRANSPARENT,
            },
        )

        frame = fig.to_image(format="png")

        frames.append(Image.open(io.BytesIO(frame)))

    frame_duration = duration // len(design_var_history)
    frames[0].save(
        fp=Path(filename).with_suffix(".gif"),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=frame_duration,
        loop=0,
    )


# def component_image_thickness_colours(
#     component_list, coords, dimensions, *plot_args, **plot_kwargs,
# ) -> go.Figure:
#     fig = go.Figure()

#     thicknesses = [c.thickness for c in component_list]
#     # colors_ids = [
#     #     {v: k for k, v in enumerate(OrderedDict.fromkeys(thicknesses))}[n]
#     #     for n in thicknesses
#     # ]
#     color_map = {0.1: 1, 0.05: 0, 0.0125: 2}
#     colors_ids = [color_map[t] for t in thicknesses]
#     color_options = COLOURS[:2] + ["#202020"]

#     sdfs = evaluate_topology_description_functions(component_list, coords)
#     contour_settings = dict(start=0, end=1, size=2)
#     colour_scales = [[[0, TRANSPARENT], [1, color_options[c]]]
#                      for c in colors_ids]

#     traces = [
#         go.Contour(
#             z=sdf.T,
#             # colorscale=colour_scales[i % len(colour_scales)],
#             colorscale=colour_scales[i],
#             contours=contour_settings,
#             line_smoothing=0,
#             showscale=False,
#             showlegend=False,
#             x=np.linspace(0, dimensions[0], coords[0].shape[0]),
#             y=np.linspace(0, dimensions[1], coords[0].shape[1]),
#             *plot_args,
#             **plot_kwargs,
#         )
#         for i, sdf in enumerate(sdfs)
#     ]

#     fig.add_traces(traces)
#     fig.update_layout(
#         dict(
#             template="simple_white",
#             plot_bgcolor=TRANSPARENT,
#             paper_bgcolor=TRANSPARENT,
#         ),
#     )

#     return fig


# def heatmap_animation(steps, duration: int = 5_000) -> go.Figure:
#     frame_duration = duration // steps.shape[2]
#     fig = go.Figure(
#         data=[go.Contour(z=steps[:, :, 0].T)],
#         layout=go.Layout(
#             title="MMC",
#             updatemenus=[
#                 dict(
#                     type="buttons",
#                     buttons=[
#                         dict(
#                             label="Play",
#                             method="animate",
#                             args=[
#                                 None, {"frame": {"duration": frame_duration}}],
#                         ),
#                     ],
#                 ),
#             ],
#             width=1_600,
#             height=800,
#         ),
#         frames=[
#             go.Frame(data=[go.Contour(z=steps[:, :, i].T)])
#             for i in range(steps.shape[2])
#         ],
#     )
#     return fig


# def save_heatmap_animation(steps, duration: int = 5_000, filename: str = "mmc") -> None:
#     frames: list[Image.Image] = []
#     for i in range(steps.shape[2]):
#         frame = go.Figure(
#             data=[go.Contour(z=steps[:, :, i].T)],
#             layout=go.Layout(
#                 width=1_600,
#                 height=800,
#             ),
#         ).to_image(format="png")
#         frames.append(Image.open(io.BytesIO(frame)))
#     frame_duration = duration // steps.shape[2]
#     frames[0].save(
#         fp=Path(filename).with_suffix(".gif"),
#         save_all=True,
#         append_images=frames[1:],
#         duration=frame_duration,
#         loop=0,
#     )


# def objective_and_constraint(objective, constraint) -> go.Figure:
#     obj_fig = make_subplots(specs=[[{"secondary_y": True}]])
#     obj_fig.add_trace(
#         go.Scatter(
#             x=np.arange(len(constraint)), y=objective, mode="lines", name="Objective",
#         ),
#         secondary_y=False,
#     )
#     obj_fig.add_trace(
#         go.Scatter(
#             x=np.arange(len(constraint)),
#             y=constraint,
#             mode="lines",
#             name="Volume Fraction Error",
#         ),
#         secondary_y=True,
#     )
#     obj_fig.update_layout(title="Objective", template="simple_white")
#     return obj_fig


# def objectives_comparison(objective: list) -> go.Figure:
#     obj_fig = go.Figure()

#     for i, obj in enumerate(objective):
#         obj_fig.add_trace(
#             go.Scatter(
#                 x=np.arange(len(obj)), y=obj, mode="lines", name=f"Objective {i}",
#             ),
#         )

#     obj_fig.update_layout(title="Objective", template="simple_white")
#     return obj_fig


# def phi_surface(phi, dimensions) -> go.Figure:
#     fig = go.Figure(
#         data=[
#             go.Surface(
#                 x=np.linspace(0, dimensions[0], phi.shape[1]),
#                 y=np.linspace(0, dimensions[1], phi.shape[0]),
#                 z=phi,
#                 contours={"z": {"show": True, "start": 0, "end": 1, "size": 2}},
#             ),
#         ],
#     )
#     fig.update_layout(
#         template="simple_white",
#         scene={
#             "xaxis": {"title": "x", "range": [0, dimensions[0]]},
#             "yaxis": {"title": "y", "range": [0, dimensions[1]]},
#             "zaxis": {"title": "phi"},
#             "aspectmode": "manual",
#             "aspectratio": {"x": 2, "y": 1, "z": 1},
#         },
#     )
#     return fig
