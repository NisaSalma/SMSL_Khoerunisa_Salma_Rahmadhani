import random
import time
from prometheus_client import Counter, Summary, start_http_server

REQUEST_COUNT = Counter('request_count_total', 'Total number of requests')
REQUEST_LATENCY = Summary('request_processing_seconds', 'Request latency in seconds')
ERROR_COUNT = Counter('error_count_total', 'Total number of errors')

def process_request():
    REQUEST_COUNT.inc()
    if random.random() < 0.1:
        ERROR_COUNT.inc()
    with REQUEST_LATENCY.time():
        time.sleep(random.random())

if __name__ == '__main__':
    start_http_server(8001)
    print("Prometheus exporter running on port 8001")
    while True:
        process_request()