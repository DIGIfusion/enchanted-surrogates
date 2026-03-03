import os
from dataclasses import dataclass, asdict
import yaml
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)

@dataclass
class RunData:
    """
    Container for data needed to restore the state of a run
    """
    batch_number: int
    depth: int
    submitted_samples: int

    def save(self, file_name: str):
        """
        Saves this RunData into a yaml file of the given file name.
        """
        with open(file_name, "w", encoding="ascii") as f:
            yaml.safe_dump(asdict(self), f)

    @staticmethod
    def load(file_name: str):
        """
        Loads a RunData instance from the given yaml file.

        Returns:
            result (RunData or None): The resulting object or None in the case of error.
        """
        if not os.path.isfile(file_name):
            return None

        try:
            with open(file_name, "r", encoding="ascii") as f:
                loaded_data = yaml.load(f, Loader=yaml.SafeLoader)
                return RunData(**loaded_data)
        except Exception as exc:
            log.debug(f"Loading file {file_name} failed: {exc}")
            return None
