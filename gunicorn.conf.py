import gunicorn
import os

bind = "0.0.0.0:"+os.environ.get("port", "5001")
workers = int(os.environ.get("workers", "1"))
timeout = int(os.environ.get("timeout", "600"))
threads = int(os.environ.get("threads", "1"))

gunicorn.SERVER = 'HiddenWebServer'