import tomli


def load_config():
    try:
        with open("config.toml", "rb") as config_file:
            return tomli.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            "config.toml not found, are you running this from the root of the repo?"
        )
