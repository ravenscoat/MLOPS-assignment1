import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(r'C:\Users\Admin\Desktop\MLOPS-assignment1'), '..')))

from app import app


import unittest
from flask import Flask
from flask.testing import FlaskClient
from app import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage(self):
        # Test if homepage loads correctly with a 200 status code
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()