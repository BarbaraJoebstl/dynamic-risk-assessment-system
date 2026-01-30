import logging
import os


class Logger:
    def __init__(self, name="risk_assesment_sys", log_file="lookilooki.log"):
        self.name = name
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self):
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


# Global logger instance
logger = Logger().logger
