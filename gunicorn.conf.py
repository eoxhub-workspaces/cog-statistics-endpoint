"""Gunicorn configuration for production deployment."""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:3000"
backlog = 2048

# Worker processes
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Prometheus metrics
def child_exit(server, worker):
    """Clean up prometheus multiprocess files on worker exit."""
    from prometheus_client import multiprocess
    multiprocess.mark_process_dead(worker.pid)
