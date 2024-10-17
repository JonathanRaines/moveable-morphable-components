import numpy as np

from moveable_morphable_components import components, layout, main, plot
from moveable_morphable_components.domain import Domain2D, Point2D


def cantilever() -> None:
    # Create the domain
    domain: Domain2D = Domain2D(dimensions=(2.0, 1.0), element_shape=(80, 40))

    # Set boundary Conditions
    fixed_dof_ids = domain.left_boundary_dof_ids()
    # Load the beam on the RHS half way up
    loaded_dof_ids = domain.coords_to_nearest_dof_ids(
        point=(domain.dims.x, domain.dims.y / 2),
    ).flatten()
    load_magnitudes = np.array([[0.0, -1.0]]).flatten()
    boundary_conditions = {
        "fixed_dof_ids": fixed_dof_ids,
        "loaded_dof_ids": loaded_dof_ids,
        "load_magnitudes": load_magnitudes,
    }

    # Generate the initial components
    beams = components.ComponentGroup(
        topology_description_function=components.uniform_beam(
            Point2D(*domain.node_coordinates),
        ),
        variable_initial=layout.grid_of_uniform_beams(
            2, 2, domain.dimensions, 0.1),
        variable_mins=np.array([0.0, 0.0, -np.pi / 2, 0.1, 0.05]),
        variable_maxes=np.array(
            [domain.dims.x, domain.dims.y, np.pi / 2, 2.0, 0.2],
        ),
    )

    design_variables, objective, constraint = main.main(
        max_iterations=100,
        domain=domain,
        boundary_conditions=boundary_conditions,
        volume_fraction_limit=0.4,
        component_list=[beams],
    )

    plot.save_component_animation(
        design_variable_history=design_variables,
        component_groups=[beams],
        domain=domain,
        filename="cantilever_example")


if __name__ == "__main__":
    cantilever()
