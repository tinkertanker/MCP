#!/usr/bin/env python3

# Let's get this party started!
from wsgiref.simple_server import make_server

import falcon
import threading
import time

# Falcon follows the REST architectural style, meaning (among
# other things) that you think in terms of resources and state
# transitions, which map to HTTP verbs.
class PeopleResource:

    def __init__(self):
        self.lock = threading.Lock()
        self.last_count = 0
        self.counts_by_hostname = {}
        self.sequence_number = (time.time() + 3) // 6

    def _update_counts_if_needed(self):
        current_sequence_number = (time.time() + 3) // 6
        if current_sequence_number > self.sequence_number:
            # update counts
            self.lock.acquire()
            print("...recalculating")
            counts = []
            for hostname, values in self.counts_by_hostname.items():
                counts.append(sum(values)/len(values))
            self.last_count = round(sum(counts)/len(counts)) if len(counts) else 0
            self.counts_by_hostname = {}
            self.sequence_number = current_sequence_number
            print(f'People: {self.last_count}')
            self.lock.release()

    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        resp.text = str(self.last_count)
        self._update_counts_if_needed()
    
    def on_post(self, req, resp):
        """Handles POST requests"""
        req_obj = req.media
        if "hostname" in req_obj and "people" in req_obj:
            print(f"Hostname: {req_obj['hostname']} People: {req_obj['people']}")
            self.counts_by_hostname.setdefault(req_obj['hostname'], []).append(int(req_obj['people']))
        self._update_counts_if_needed()
        resp.status = falcon.HTTP_200  # This is the default status
        resp.content_type = falcon.MEDIA_TEXT  # Default is JSON, so override
        resp.text = str(self.last_count)


# falcon.App instances are callable WSGI apps
# in larger applications the app is created in a separate file
app = falcon.App()

# Resources are represented by long-lived class instances
people = PeopleResource()

# people will handle all requests to the '/people' URL path
app.add_route('/people', people)

if __name__ == '__main__':
    with make_server('', 8000, app) as httpd:
        print('Serving on port 8000...')

        # Serve until process is killed
        httpd.serve_forever()