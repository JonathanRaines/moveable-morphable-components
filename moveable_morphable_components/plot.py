import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.express.colors import qualitative
from plotly.subplots import make_subplots

from moveable_morphable_components.main import evaluate_signed_distance_functions

COLOURS = qualitative.Pastel[:2]
TRANSPARENT = "rgba(0,0,0,0)"


def component_image(component_list, coords, dimensions) -> go.Figure:
    fig = go.Figure()

    sdfs = evaluate_signed_distance_functions(component_list, coords)
    contour_settings = dict(start=0, end=1, size=2)
    colour_scales = [[[0, TRANSPARENT], [1, c]] for c in COLOURS]

    traces = [
        go.Contour(
            z=sdf.T,
            colorscale=colour_scales[i % len(colour_scales)],
            contours=contour_settings,
            line_smoothing=0,
            showscale=False,
            showlegend=False,
            x=np.linspace(0, dimensions[0], coords[0].shape[0]),
            y=np.linspace(0, dimensions[1], coords[0].shape[1]),
        )
        for i, sdf in enumerate(sdfs)
    ]

    fig.add_traces(traces)
    fig.update_layout(
        dict(
            template="simple_white",
            plot_bgcolor=TRANSPARENT,
            paper_bgcolor=TRANSPARENT,
        )
    )

    return fig


def save_component_animation(
    component_history, coords, dimensions, duration: int = 5_000, filename: str = "mmc"
) -> None:
    frames: list[Image.Image] = []
    for component_list in component_history:
        fig = go.Figure()

        sdfs = evaluate_signed_distance_functions(component_list, coords)
        contour_settings = dict(start=0, end=1, size=2)
        colour_scales = [[[0, TRANSPARENT], [1, c]] for c in COLOURS]

        traces = [
            go.Contour(
                z=sdf.T,
                colorscale=colour_scales[i % len(colour_scales)],
                contours=contour_settings,
                line_smoothing=0,
                showscale=False,
                showlegend=False,
                x=np.linspace(0, dimensions[0], coords[0].shape[0]),
                y=np.linspace(0, dimensions[1], coords[0].shape[1]),
            )
            for i, sdf in enumerate(sdfs)
        ]

        fig.add_traces(traces)
        fig.update_layout(
            dict(
                template="simple_white",
                # plot_bgcolor=TRANSPARENT, #TODO(JonathanRaines): PIL seems to layer images so can't have transparent background
                # paper_bgcolor=TRANSPARENT,
            )
        )

        frame = fig.to_image(format="png")

        frames.append(Image.open(io.BytesIO(frame)))

    frame_duration = duration // len(component_history)
    frames[0].save(
        fp=Path(filename).with_suffix(".gif"),
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=frame_duration,
        loop=0,
    )


def component_image_thickness_colours(
    component_list, coords, dimensions, *plot_args, **plot_kwargs
) -> go.Figure:
    fig = go.Figure()

    thicknesses = [c.thickness for c in component_list]
    # colors_ids = [
    #     {v: k for k, v in enumerate(OrderedDict.fromkeys(thicknesses))}[n]
    #     for n in thicknesses
    # ]
    color_map = {0.1: 1, 0.05: 0, 0.0125: 2}
    colors_ids = [color_map[t] for t in thicknesses]
    color_options = COLOURS[:2] + ["#202020"]

    sdfs = evaluate_signed_distance_functions(component_list, coords)
    contour_settings = dict(start=0, end=1, size=2)
    colour_scales = [[[0, TRANSPARENT], [1, color_options[c]]] for c in colors_ids]

    traces = [
        go.Contour(
            z=sdf.T,
            # colorscale=colour_scales[i % len(colour_scales)],
            colorscale=colour_scales[i],
            contours=contour_settings,
            line_smoothing=0,
            showscale=False,
            showlegend=False,
            x=np.linspace(0, dimensions[0], coords[0].shape[0]),
            y=np.linspace(0, dimensions[1], coords[0].shape[1]),
            *plot_args,
            **plot_kwargs,
        )
        for i, sdf in enumerate(sdfs)
    ]

    fig.add_traces(traces)
    fig.update_layout(
        dict(
            template="simple_white",
            plot_bgcolor=TRANSPARENT,
            paper_bgcolor=TRANSPARENT,
        )
    )

    return fig


def heatmap_animation(steps, duration: int = 5_000) -> go.Figure:
    frame_duration = duration // steps.shape[2]
    fig = go.Figure(
        data=[go.Contour(z=steps[:, :, 0].T)],
        layout=go.Layout(
            title="MMC",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": frame_duration}}],
                        )
                    ],
                )
            ],
            width=1_600,
            height=800,
        ),
        frames=[
            go.Frame(data=[go.Contour(z=steps[:, :, i].T)])
            for i in range(steps.shape[2])
        ],
    )
    return fig


def save_heatmap_animation(steps, duration: int = 5_000, filename: str = "mmc") -> None:
    frames: list[Image.Image] = []
    for i in range(steps.shape[2]):
        frame = go.Figure(
            data=[go.Contour(z=steps[:, :, i].T)],
            layout=go.Layout(
                width=1_600,
                height=800,
            ),
        ).to_image(format="png")
        frames.append(Image.open(io.BytesIO(frame)))
    frame_duration = duration // steps.shape[2]
    frames[0].save(
        fp=Path(filename).with_suffix(".gif"),
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
    )


def objective_and_constraint(objective, constraint) -> go.Figure:
    obj_fig = make_subplots(specs=[[{"secondary_y": True}]])
    obj_fig.add_trace(
        go.Scatter(
            x=np.arange(len(constraint)), y=objective, mode="lines", name="Objective"
        ),
        secondary_y=False,
    )
    obj_fig.add_trace(
        go.Scatter(
            x=np.arange(len(constraint)),
            y=constraint,
            mode="lines",
            name="Volume Fraction Error",
        ),
        secondary_y=True,
    )
    obj_fig.update_layout(title="Objective", template="simple_white")
    return obj_fig


def objectives_comparison(objective: list) -> go.Figure:
    obj_fig = go.Figure()

    for i, obj in enumerate(objective):
        obj_fig.add_trace(
            go.Scatter(
                x=np.arange(len(obj)), y=obj, mode="lines", name=f"Objective {i}"
            ),
        )

    obj_fig.update_layout(title="Objective", template="simple_white")
    return obj_fig


def phi_surface(phi, dimensions) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Surface(
                x=np.linspace(0, dimensions[0], phi.shape[1]),
                y=np.linspace(0, dimensions[1], phi.shape[0]),
                z=phi,
                contours={"z": {"show": True, "start": 0, "end": 1, "size": 2}},
            )
        ],
    )
    fig.update_layout(
        template="simple_white",
        scene={
            "xaxis": {"title": "x", "range": [0, dimensions[0]]},
            "yaxis": {"title": "y", "range": [0, dimensions[1]]},
            "zaxis": {"title": "phi"},
            "aspectmode": "manual",
            "aspectratio": {"x": 2, "y": 1, "z": 1},
        },
    )
    return fig
