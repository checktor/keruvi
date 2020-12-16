from datetime import datetime
import requests
from urllib.parse import urljoin
import uuid

import numpy
from tensorflow.keras.callbacks import Callback

class KeruviRemoteMonitor(Callback):

    def __init__(self, root, path="", model_id="", use_batch_callback=False, use_epoch_callback=True, use_train_callback=False):
        # Create URLs to REST endpoints
        # handling 'on_batch', 'on_epoch'
        # and 'on_train' Keras callbacks.
        self.url = urljoin(root, path)
        self.batch_url = urljoin(self.url, "batch")
        self.epoch_url = urljoin(self.url, "epoch")
        self.train_url = urljoin(self.url, "train")
        # Use provided 'model_id' as unique
        # identifier for monitored Keras run.
        # If it is not provided, create a
        # randomly chosen UUID.
        if (model_id):
            self.id = model_id
        else:
            self.id = uuid.uuid4()
        self.use_batch_callback = use_batch_callback
        self.use_epoch_callback = use_epoch_callback
        self.use_train_callback = use_train_callback

    def _create_payload(self, logs):
        converted_logs = {}
        for key, value in logs.items():
            if (isinstance(value, numpy.float32) or isinstance(value, numpy.float64)):
                # Convert NumPy specific floating point
                # numbers to Python's float data type.
                converted_logs[key] = float(value)
            else:
                converted_logs[key] = value
        return {
            "id": self.id,
            "metrics": {
                "timestamp": datetime.now().isoformat(),
                "logs": converted_logs
            }
        }
        
    def get_model_id(self):
        return self.id
        
    def on_batch_end(self, batch, logs={}):
        if (self.use_batch_callback):
            payload = self._create_payload(logs)
            requests.post(self.batch_url, json=payload)
    
    def on_epoch_end(self, epoch, logs={}):
        if (self.use_epoch_callback):
            payload = self._create_payload(logs)
            requests.post(self.epoch_url, json=payload)

    def on_train_end(self, logs={}):
        if (self.use_train_callback):
            payload = self._create_payload(logs)
            requests.post(self.train_url, json=payload)
