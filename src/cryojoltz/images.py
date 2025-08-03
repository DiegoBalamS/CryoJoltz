import cryojax.simulator as cxs
from cryojax.constants import (
    get_tabulated_scattering_factor_parameters,
    read_peng_element_scattering_factor_parameter_table,
)
from cryojax.ndimage import downsample_with_fourier_cropping
import jax.numpy as jnp

def simulate_image_from_atoms(atom_positions, atom_types, shape=(240, 240, 240), voxel_size=1.0, downsampling_factor=3):
    """
    Generates a simulated cryo-EM-style image from atomic positions and types.

    Parameters:
    -----------
    atom_positions : array (N, 3)
        3D coordinates of the atoms (in Ångströms).
    atom_types : list of str
        List of chemical elements (e.g., ["C", "N", "O", ...]).
    shape : tuple
        Size of the 3D grid.
    voxel_size : float
        Voxel size in Ångströms.
    downsampling_factor : int
        Factor to reduce resolution using Fourier cropping.

    Returns:
    --------
    integrated_potential : array (H, W)
        2D simulated image obtained by projecting the atomic potential.
    """

    #Obtain scattering parameters
    scatter_params = get_tabulated_scattering_factor_parameters(
        atom_types,
        read_peng_element_scattering_factor_parameter_table())

    # Create atomic potential
    atom_potential = cxs.PengAtomicPotential(
        atom_positions,
        scattering_factor_a=scatter_params["a"],
        scattering_factor_b=scatter_params["b"],)

    # Evaluate potential on a 3D grid
    real_voxel_grid = atom_potential.as_real_voxel_grid(
        shape,
        voxel_size,
        batch_size_for_z_planes=1,)

    # Reduce resolution
    downsampled_voxel_grid = downsample_with_fourier_cropping(
        real_voxel_grid,
        downsampling_factor,)
    downsampled_voxel_size = downsampling_factor * voxel_size

    # Convert to Fourier potential
    voxel_potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
        downsampled_voxel_grid, downsampled_voxel_size)

    # Project the potential to 2D
    integrator = cxs.FourierSliceExtraction()  # <- si usas equinox ≥ 0.11.x
    config = cxs.BasicConfig(
        shape=voxel_potential.shape[0:2],
        pixel_size=voxel_potential.voxel_size,
        voltage_in_kilovolts=300.0,)
    integrated_potential = integrator.integrate(
        voxel_potential,
        config,
        outputs_real_space=True,)

    return integrated_potential
