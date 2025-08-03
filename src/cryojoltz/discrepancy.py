import jax.numpy as jnp
import jax

"""Maximum Mean Discrepancy (MMD) polynomial kernel"""
@jax.jit
def polinomial(dot,alpha,c,d):
    return (alpha*dot+c)**d

@jax.jit
def diff_polinomial_jax(dot,alpha,c,d,y):
    cons=alpha*d*(alpha*dot+c)**(d-1)
    return jnp.matmul(cons, y)

@jax.jit
def mmd_polinomial_kernel_jax(x,y,alpha,c,d):
    x=x/1000
    y=y/1000
    dot_1 = jnp.matmul(x, y.T)
    dot_2 = jnp.matmul(x, x.T)
    dot_3 = jnp.matmul(y, y.T)
    return -(polinomial(dot_1,alpha,c,d).sum(axis=1)+polinomial(dot_2,alpha,c,d).sum(axis=1)/2)+(polinomial(dot_3,alpha,c,d).sum(axis=1)/2)

@jax.jit
def diff_polinomial_kernel_jax(x,y,alpha,c,d):
    dot_1 = jnp.matmul(x, y.T)
    dot_2 = jnp.matmul(x, x.T)
    return -diff_polinomial_jax(dot_1,alpha,c,d,y)+diff_polinomial_jax(dot_2,alpha,c,d,x)

"""Energy Distance"""

@jax.jit
def distance(x, y):
    # Compute ||x||^2 and ||y||^2 in a vectorized way.
    x_norm_sq = jnp.sum(x**2, axis=1, keepdims=True)  # (1000, 1)
    y_norm_sq = jnp.sum(y**2, axis=1, keepdims=True).T  # (1, 1207)

    # Compute the dot product x · y^T
    cross_term = jnp.dot(x, y.T)

    # Compute the squared distances.
    dist_sq = x_norm_sq + y_norm_sq - 2 * cross_term

    # Take the square root to obtain the distances and compute the total sum
    dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-8))  # Usa jnp.maximum para evitar valores negativos debido a redondeo
    total_sum = jnp.sum(dist)
    
    return total_sum

@jax.jit
def distance_normalize(x_single,y):
    dif=x_single[:, None,:]-y[None,:,:]
    norms=jnp.linalg.norm(dif,axis=-1,keepdims=True)
    norms = jnp.where(norms == 0, 1, norms)
    dif_norm=dif/norms
    sum_total = jnp.sum(dif_norm,axis=1)
    return sum_total 

@jax.jit
def energy_distance(x,y,n):
    energy_dist=2*distance(x,y)/(n**2)-distance(x,x)/(n**2)-distance(y,y)/(n**2)
    return energy_dist

@jax.jit
def grad_energy_distance(x,y,n):
    grad_x = jax.grad(energy_distance, argnums=0)
    grad_x_value = grad_x(x, y,n)
    return grad_x_value

"""Cross-entropy"""

