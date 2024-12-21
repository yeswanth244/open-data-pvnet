import yaml


def load_config(config_path: str):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Usage
config = load_config("src/open_data_pvnet/configs/met_office_data_config.yaml")
print(config["input_data"]["nwp"]["met_office"]["nwp_channels"])
