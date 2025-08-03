import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from cryojoltz.images import simulate_image_from_atoms
from cryojoltz.discrepancy import image_distance

def loss_fn(model, features, key, target_images):
    pred = model(features, key=key, sample_structure=True)
    coords = pred["sample_atom_coords"][0]
    atom_types_onehot = features["ref_element"][0]
    identities = jnp.argmax(atom_types_onehot, axis=-1)
    generated = simulate_image_from_atoms(coords,identities)
    return image_distance(generated, target_images)

grad_fn = eqx.filter_value_and_grad(loss_fn)
