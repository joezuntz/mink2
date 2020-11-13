import numpy as np
import tempfile
import contextlib
import os
import subprocess
import pyccl
from astropy.io import fits
import healpy


# __file__ is automatically defined in every python file
# as the name of the current file.  This function extracts
# the directory part of the file name.
dirname = os.path.dirname(__file__)
# This one gets the absolute path to the example_config file
# in the same directory as this file.  Absolute means it has the full
# /path/to/this/directory, so even if we change directories we
# can still open the file
template_file = os.path.abspath(os.path.join(dirname, "example.config"))
des_file = os.path.abspath(os.path.join(dirname, "y1_redshift_distributions_v1.fits"))


# This is a little python thing that lets you run
# commands in a temporary directory, so that if we
# generate a bunch of files they won't clutter thngs up.
# We use it later on.
# The @ sign is called a "decorator"
@contextlib.contextmanager
def inside_temp_directory():
    # make a temporary directory somewhere.
    # the with statement means that at the end
    # of this the directory will be deleted
    with tempfile.TemporaryDirectory() as dirname:
        # get the directory we're to start with
        orig_dir = os.getcwd()
        # change to that directory
        os.chdir(dirname)
        # this makes it so whatever happens
        # (even if there is a crash or something)
        # we will change back to the original directory
        try:
            # this yield thing means we now go and run
            # code outside this function, inside the
            # "with" block below.
            yield dirname
        finally:
            os.chdir(orig_dir)


def apply_normalization(z, n_of_z, n_total):
    """Re-normalize our galaxy sample so it represents n_total objects

    Our galaxy sample is characterised here by n(z), the histogram
    of the redshifts of the galaxies.  To start with it's normalized
    to integrate to 1, like a probability.  But FLASK needs it to be
    normalized so that the area under the curve represents the number
    of galaxies per square arcminute, over all redshifts.

    So we have to re-scale it.
    """
    output = []
    # for each tomographic bin:
    for nz, nt in zip(n_of_z, n_total):
        # get the current area under the curve
        norm = np.trapz(nz, z)
        # rescale to the new area
        nz = nz / norm * nt
        # record the output
        output.append(nz)
    return output


def write_fields(z, lens_n_of_z, source_n_of_z):
    """Create two data files that FLASK needs, n(z) and fields info.


    We have hard-coded in the template that the field file is called fields.dat
    and the n(z) files (which are only needed for the lens samples) are called
    n_of_z-f0.dat
    n_of_z-f1.dat
    etc.

    We create these files now.
    """
    f = open("fields.dat", "w")

    n_lens = len(lens_n_of_z)
    n_source = len(source_n_of_z)

    for i in range(n_lens):
        nz = lens_n_of_z[i]

        # Record zmin, zmax
        z_non_zero = z[nz != 0]
        zmin = z_non_zero.min()
        zmax = z_non_zero.max()

        # This is always 1 for clustering
        shift = 1.0
        f.write(f"{i}     {i}    0.0    {shift}    1    {zmin}   {zmax}\n")

        # and save the n(z) file.
        nz_data = np.vstack((z, nz)).T
        np.savetxt(f"n_of_z-f{i}.dat", nz_data)

    # Same for sources
    for i in range(n_source):
        # flask is a bit weird with numbering - we have
        # to number source and lens fields separately
        k = i + n_lens
        # n(z) calculation
        nz = source_n_of_z[i]
        z_non_zero = z[nz != 0]
        zmin = z_non_zero.min()
        zmax = z_non_zero.max()

        # Get the mean z, which we need to work out the shift parameter
        mean_z = np.trapz(lens_n_of_z[i] * z, z) / np.trapz(lens_n_of_z[i], z)
        # print(f"Mean z_{i} = {mean_z}")
        shift = XavierShift(mean_z)
        f.write(f"{k}     {k}   0.0    {shift}    2    {zmin}   {zmax}\n")

    f.close()


def compute_cl(cosmo_params, biases, z, lens_n_of_z, source_n_of_z):
    cosmo = pyccl.Cosmology(**cosmo_params)
    ell = np.arange(2, 2001)
    field = 0  # lenses

    # Construct lens "tracer" objects that we will use to
    # compute the spectra.  These contain information about the galaxies
    # that DES observed.
    tracers = []

    # First the lenses
    for i, (lens_nz, bias) in enumerate(zip(lens_n_of_z, biases)):
        tracer = pyccl.tracers.NumberCountsTracer(
            cosmo, has_rsd=False, dndz=(z, lens_nz), bias=(z, np.repeat(bias, z.size))
        )
        tracers.append((field + i, tracer))

    field = len(lens_n_of_z)
    # Next the sources
    for i, source_nz in enumerate(source_n_of_z):
        tracer = pyccl.tracers.WeakLensingTracer(cosmo, dndz=(z, source_nz))
        tracers.append((field + i, tracer))

    # Now we have the tracers we compute the spectra between each pair of them
    c_ell = {"ell": ell}
    for field1, tracer1 in tracers[:]:
        for field2, tracer2 in tracers[:]:
            # we avoid calculating the same thing twice
            if (field2, field1) in c_ell:
                continue
            # use CCL to do the calculation
            cl = pyccl.angular_cl(cosmo, tracer1, tracer2, ell)
            # and store it in a dictionary.
            c_ell[field1, field2] = cl

    return c_ell


def write_cl(c_ell):
    """Save our measured spectra C_ell to the files FLASK
    will read them from.

    """
    # Pull out the ell values
    ell = c_ell["ell"]
    for tags, cl in c_ell.items():
        # Skip "ell" itself
        if tags == "ell":
            continue

        # Construct the name of the file name
        field1, field2 = tags
        filename = f"C_ell-f{field1}z{field1}f{field2}z{field2}.dat"

        # Combine and save ell and C_ell to the two columns
        data = np.vstack((ell, cl))
        np.savetxt(filename, data.T)


def run_flask(
    cosmo_params, nside, z, lens_n_of_z, source_n_of_z, c_ell, smoothing, source_n_total, seed
): #add numpy seed input

    np.random.seed(seed)
    
    # First save out C_ell values to input files
    write_cl(c_ell)

    # and then write out other info flask will need
    write_fields(z, lens_n_of_z, source_n_of_z)

    # pick a random seed
    flask_seed = np.random.randint(1000000000)

    # read a template version of the configuration file for FLASK
    template = open(template_file, "r").read()

    nbin_source = len(source_n_of_z)
    nbin_lens = len(lens_n_of_z)

    # This is the intrinsic scatter in the ellipticity
    sigma_e = 0.26

    # Matter density.
    omega_m = cosmo_params["Omega_c"] + cosmo_params["Omega_b"]

    # create the configuration file that FLASK operates on
    config = template.format(
        seed=flask_seed, omega_m=omega_m, omega_l=1 - omega_m, sigma_e=sigma_e, nside=nside,
    )

    # write config
    open("flask.config", "w").write(config)

    # run flask
    subprocess.check_call("flask flask.config".split())

    # read the maps flask has created from files
    clustering_maps = [
        healpy.read_map(f"clustering-map-f{i}z{i}.fits", verbose=False, dtype=None).astype(np.float64)
        for i in range(nbin_lens)
    ]
    convergence_maps = [
        healpy.read_map(f"lensing-map-f{i+nbin_lens}z{i+nbin_lens}.fits", verbose=False, dtype=None).astype(np.float64)
        for i in range(nbin_source)
    ]

    # add noise.  FLASK already has noise in the clustering maps, so we
    # only have to add it to the convergence maps
    for i in range(nbin_source):  # work out the noise for this bin
        A_pix_arcmin = healpy.nside2pixarea(nside, degrees=True) * 60 ** 2
        n_gal_pixel = source_n_total[i] * A_pix_arcmin
        sigma_pixel = sigma_e / np.sqrt(n_gal_pixel)

        # and add it to this map
        noise = np.random.normal(size=convergence_maps[i].size) * sigma_pixel
        convergence_maps[i] += noise

    # smooth the maps.  We have to convert the smoothing value from arcmin
    # to radians.  verbose=False stops it from printing out loads of stuff
    for i in range(nbin_lens):
        clustering_maps[i] = healpy.smoothing(
            clustering_maps[i], fwhm=np.radians(smoothing / 60), verbose=False, iter=1
        )

    for i in range(nbin_source):
        convergence_maps[i] = healpy.smoothing(
            convergence_maps[i], fwhm=np.radians(smoothing / 60), verbose=False, iter=1
        )

    return clustering_maps, convergence_maps


def simulate_maps(
    cosmo_params,
    nside,
    biases,
    z,
    lens_n_of_z,
    lens_n_total,
    source_n_of_z,
    source_n_total,
    smoothing,
    seed,
    nmax=None,
):
    # normalize the n(z) to the correct density
    lens_n_of_z = apply_normalization(z, lens_n_of_z, lens_n_total)
    source_n_of_z = apply_normalization(z, source_n_of_z, source_n_total)

    # Just simulate a reduced number of maps
    if nmax is not None:
        lens_n_of_z = lens_n_of_z[:nmax]
        source_n_of_z = source_n_of_z[:nmax]
        lens_n_total = lens_n_total[:nmax]
        source_n_total = source_n_total[:nmax]
        biases = biases[:nmax]

    with inside_temp_directory() as tmpdir:
        # Compute power spectra C_ell
        c_ell = compute_cl(cosmo_params, biases, z, lens_n_of_z, source_n_of_z)

        # and run flask on them to compute maps
        maps = run_flask(
            cosmo_params,
            nside,
            z,
            lens_n_of_z,
            source_n_of_z,
            c_ell,
            smoothing,
            source_n_total,
            seed,
        )

    return maps


def simulate_des_maps(omega_m, sigma_8, smoothing, nside, nmax=None, seed=29101995):
    f = fits.open(des_file)
    source_n_of_z = []
    source_n_total = []

    # Load DES source (convergence) sample from the FITS
    # file
    ext = f["nz_source_mcal"]
    hdr = ext.header
    z = ext.data["Z_MID"][:]
    for b in range(1, 5):
        nz = ext.data[f"BIN{b}"][:]
        source_n_of_z.append(nz)
        ngal = hdr[f"NGAL_{b}"]
        source_n_total.append(ngal)

    # and the lens (clustering) sample from the same file.
    lens_n_of_z = []
    lens_n_total = []

    ext = f["nz_lens"]
    hdr = ext.header
    z = ext.data["Z_MID"][:]
    for b in range(1, 6):
        nz = ext.data[f"BIN{b}"][:]
        lens_n_of_z.append(nz)
        ngal = hdr[f"NGAL_{b}"]
        lens_n_total.append(ngal)

    f.close()

    # construct the dictionary of parameters
    # we will need.  We fix some and compute others.
    cosmo_params = {
        "Omega_b": 0.048,
        "Omega_c": omega_m - 0.048,
        "h": 0.7,
        "n_s": 0.96,
        "sigma8": sigma_8,
    }

    # These are galaxy biases, the ratio between dark matter density
    # and galaxy density.  In real analyses we would need to vary these
    # but for now we fix them.
    biases = [1.42, 1.65, 1.60, 1.92, 2.00]

    return simulate_maps(
        cosmo_params,
        nside,
        biases,
        z,
        lens_n_of_z,
        lens_n_total,
        source_n_of_z,
        source_n_total,
        smoothing,
        seed,
        nmax=nmax,  
    )


# This calculates one of the parameters that is needed by the
# FLASK code
def XavierShift(z):
    a0 = 0.2
    s0 = 0.568591
    return a0 * (((z * s0) ** 2 + z * s0 + 1) / (z * s0 + 1) - 1)





## Added by Nisha to test galaxy bias ##
    
def simulate_des_maps_bias(omega_m, sigma_8, smoothing, nside, b1, nmax=None, seed=29101995):
    f = fits.open(des_file)
    source_n_of_z = []
    source_n_total = []

    # Load DES source (convergence) sample from the FITS
    # file
    ext = f["nz_source_mcal"]
    hdr = ext.header
    z = ext.data["Z_MID"][:]
    for b in range(1, 5):
        nz = ext.data[f"BIN{b}"][:]
        source_n_of_z.append(nz)
        ngal = hdr[f"NGAL_{b}"]
        source_n_total.append(ngal)

    # and the lens (clustering) sample from the same file.
    lens_n_of_z = []
    lens_n_total = []

    ext = f["nz_lens"]
    hdr = ext.header
    z = ext.data["Z_MID"][:]
    for b in range(1, 6):
        nz = ext.data[f"BIN{b}"][:]
        lens_n_of_z.append(nz)
        ngal = hdr[f"NGAL_{b}"]
        lens_n_total.append(ngal)

    f.close()

    # construct the dictionary of parameters
    # we will need.  We fix some and compute others.
    cosmo_params = {
        "Omega_c": omega_m - 0.049,
        "Omega_b": 0.048,
        "h": 0.7,
        "n_s": 0.96,
        "sigma8": sigma_8,
    }

    # These are galaxy biases, the ratio between dark matter density
    # and galaxy density.  In real analyses we would need to vary these
    # but for now we fix them.
    biases = [b1, 1.65, 1.60, 1.92, 2.00]

    return simulate_maps(
        cosmo_params,
        nside,
        biases,
        z,
        lens_n_of_z,
        lens_n_total,
        source_n_of_z,
        source_n_total,
        smoothing,
        seed,
        nmax=nmax,   
    )
