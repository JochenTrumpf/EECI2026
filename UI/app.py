from abc import ABC
from dash import Dash

from UI.HTTP_server import CustomTCPServer, CrossOriginHTTPRequestHandler
from threading import Thread

from .unity_player import unity_player

class dash_app(ABC):
    """
    Abstract baseclass for all dash app objects. 
    
    Callable interface:
     - __init__(url_prefix, supply_HTTP_download_server) 
        + url_prefix: pathname prefix for dash requests (default: '/' = current folder) 
        + supply_HTTP_server: if set to true the app will start a local HTTP server for single run downloads
        + HTTP_download_url: location of the single run download file
        + use_unity: if set to true the app will start a unity player for 3D scenarios

    Re-implement __init__() in any subclass to add aditional app parameters (if applicable)
    """

    def __init__(self, url_prefix: str = '/', supply_HTTP_server: bool = True, HTTP_download_url: str = 'http://localhost:8000/single_run', use_unity: bool = True):

        self.app = Dash(requests_pathname_prefix = url_prefix)
        
        if supply_HTTP_server:
            self.HTTP_download_server = CustomTCPServer(('', 8000), CrossOriginHTTPRequestHandler)

            self.HTTP_download_server_thread = Thread(target = self.run_HTTP_download_server, daemon = True)
            self.HTTP_download_server_thread.start()

            self.HTTP_download_url = HTTP_download_url

        if use_unity:
            self.unity_player = unity_player(self, supply_HTTP_server)

    def run_HTTP_download_server(self):
        self.HTTP_download_server.serve_forever()

    def _assemble_unity_response(self, request):
        response = {
            'type': 'error',
            'message': 'dash_app base class implementation of _assemble_unity_response() called in response to request {}.'.format(request)
        }
        return response
