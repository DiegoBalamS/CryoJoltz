import cryojax.simulator as cxs
from cryojax.constants import (
    get_tabulated_scattering_factor_parameters,
    read_peng_element_scattering_factor_parameter_table,
)
from cryojax.ndimage import downsample_with_fourier_cropping
import jax.numpy as jnp

def simulate_image_from_atoms(atom_positions, atom_types, pixel_size=1.0):
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
    potential = cxs.PengAtomicPotential(
        atom_positions,
        scattering_factor_a=scatter_params["a"],
        scattering_factor_b=scatter_params["b"],)

    # Evaluate potential on a 3D grid
    ctf=cxs.AberratedAstigmaticCTF(defocus_in_angstroms=10000.0,
    astigmatism_in_angstroms=-100.0,
    astigmatism_angle=10.0,
)
    transfer_theory = cxs.ContrastTransferTheory(ctf, amplitude_contrast_ratio=0.1)
    config = cxs.BasicConfig((128, 128), pixel_size, 300, pad_options = dict(shape=(128, 128)))

    image_model = cxs.make_image_model(
    potential,
    config,
    pose,
    transfer_theory,
    normalizes_signal=True,)

    return image_model.simulate()
