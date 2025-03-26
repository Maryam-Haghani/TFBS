# mylogger.py
import os
import logging


class CustomLogger:
    def __init__(self, name: str, log_directory: str = '.', log_file: str = 'app.log', level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._configure_handlers(log_directory, log_file, level)

    def _configure_handlers(self, log_directory: str, log_file: str, level: int):
        # Clear existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Ensure the log directory exists
        os.makedirs(log_directory, exist_ok=True)

        # Build the full path for the log file
        log_path = os.path.join(log_directory, log_file)

        # Create a file handler that writes to the specified log file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)

        # Create a console (stream) handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Define a formatter and add it to both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add both handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger