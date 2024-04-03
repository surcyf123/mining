import os
import csv
import time

class CSVLogger:
    def __init__(self, filename, fields):
        self.filename = filename
        self.fields = fields
        
        # Check if the file exists and is empty, then write headers
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
            with open(self.filename, mode='a', newline='') as file:  # Use 'a' mode here
                writer = csv.DictWriter(file, fieldnames=self.fields)
                writer.writeheader()

    def log(self, **kwargs):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            log_data = {"Timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}
            log_data.update(kwargs)
            writer.writerow(log_data)
