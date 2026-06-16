import pytest
from pathlib import Path
from enchanted_surrogates.supervisor.supervisor import Supervisor
from enchanted_surrogates.utils.logger import LoggerConfig
from run import load_configuration, setup_logger

@pytest.fixture
def run_config(tmp_path, request):
    """
    Factory fixture to setup and run the program from the given config file.
    """
    def _run(config_file: str, call_start: bool = True):
        test_dir = Path(request.fspath).parent
        config_path = (test_dir / config_file).resolve()

        args = load_configuration(str(config_path))
        args.supervisor["base_run_dir"] = tmp_path

        supervisor = Supervisor(args, config_path=config_path)
        setup_logger(supervisor.base_run_dir, "INFO")
        
        if (call_start):
            supervisor.start()

        return supervisor

    yield _run
    # Teardown
    LoggerConfig.reset()
