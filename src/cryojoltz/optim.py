# train_joltz.py

import jax
import jax.numpy as jnp
import optax
import equinox as eqx


def train_joltz_model(joltz_model, jax_features, target_images, loss_fn, learning_rate=1e-4, steps=4):
    """
    Trains the Joltz model using a given loss function and target images.

    Parameters:
        joltz_model: Equinox model returned by joltz.from_torch.
        jax_features: Input features in JAX format.
        target_images: Target images to compare against.
        loss_fn: A callable loss function of the form loss_fn(model, features, key, target_images).
        learning_rate: Learning rate for the optimizer.
        steps: Number of optimization steps to run.

    Returns:
        A fine-tuned Equinox model with updated parameters.
    """
    trainable_params, static_parts = eqx.partition(joltz_model, eqx.is_inexact_array)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(trainable_params)

    for step in range(steps):
        key = jax.random.PRNGKey(step)

        def loss_fn_wrapped(trainable_params):
            model = eqx.combine(trainable_params, static_parts)
            return loss_fn(model, jax_features, key, target_images)

        loss, grads = jax.value_and_grad(loss_fn_wrapped)(trainable_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable_params = optax.apply_updates(trainable_params, updates)

        print(f"[{step}] Loss: {loss:.5f}")

    return eqx.combine(trainable_params, static_parts)

