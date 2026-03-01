import pytest
from enchanted_surrogates.supervisor.supervisor import Supervisor
from run import load_configuration, setup_logger

@pytest.fixture
def run_config(config_file: str) -> Supervisor:
    """
    Helper fixture to run the program with the given config file.
    """
    args = load_configuration(config_file)
    supervisor = Supervisor(args, config_path=config_file)

    setup_logger(supervisor.base_run_dir, "INFO")

    supervisor.start()
    return supervisor