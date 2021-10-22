from cosmosis.datablock import option_section



def setup(options):

    a_type = options[option_section, "a_type"]
    return {
        "a_type": a_type,
    }


def execute(block, config):


    a_type = config["a_type"]
    # config is whatever came from the setup function
    omega_m = block["cosmological_parameters", "omega_m"]
    # etc.


    L = ...

    block["likelihoods", "mfcl_like"] = L


    return 0