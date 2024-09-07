import jax
import jax.numpy as jnp
import numpy as np

from moveable_morphable_components.components import CircleSpec, Point2D, circle
from moveable_morphable_components.domain import Domain2D
from moveable_morphable_components.main import plot_values

domain = Domain2D(dimensions=(2.0, 1.0), element_shape=(80, 40))
coords = [dim.flatten(order="F") for dim in domain.node_coordinates]


circ, J = circle(Point2D(*coords))
spec = CircleSpec(center=Point2D(1.0, 0.0), radius=1.0)
circ_grad: CircleSpec = J(spec)
# plot_values(circ(spec), domain.node_shape).show()
# plot_values(circ_grad.center.x, domain.node_shape).show()
# plot_values(circ_grad.center.y, domain.node_shape).show()
# plot_values(circ_grad.radius, domain.node_shape).show()
design_vars = np.array(
    [
        [0.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    ]
).T

specs = CircleSpec(center=Point2D(*design_vars[:2]), radius=design_vars[2])

plot_values(jnp.max(jax.vmap(circ)(specs), axis=0), domain.node_shape).show()
circ_grads = jax.vmap(J)(specs)
plot_values(circ_grads.center.x[0], domain.node_shape).show()
plot_values(circ_grads.center.x[1], domain.node_shape).show()
