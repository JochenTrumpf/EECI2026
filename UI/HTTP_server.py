import socketserver

import http.server
from http import HTTPStatus

import os

# make a custom HTTP server that can handle a suggested filename for downloads
class CustomTCPServer(socketserver.TCPServer):

    def __init__(self, server_address, RequestHandlerClass):
        self.suggested_filename: str
        super().__init__(server_address, RequestHandlerClass)

# this implementation removes A LOT of functionality from the base class implementation
# it is ONLY suitable for serving local binary files with a valid path given
class CrossOriginHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, request, client_address, server: CustomTCPServer):
        self.server: CustomTCPServer
        
        # call base class method
        super().__init__(request, client_address, server)
    
    def send_head(self):
        path = self.translate_path(self.path)
        f = open(path, 'rb')
        fs = os.fstat(f.fileno())
        self.send_response(HTTPStatus.OK)
        # these are the added headers that enable cross-origin requests to download a file
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header(
            'Content-Disposition', 
            'attachment; filename="{}"'.format(self.server.suggested_filename))
        self.send_header('Content-Type', 'application/octet-stream')
        # the rest is as in the base class implementation
        self.send_header('Content-Length', str(fs[6]))
        self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

# this implementation is ONLY suitable for serving unity files with a valid path given
class UnityHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):

    def send_head(self):
        path = self.translate_path(self.path)
        # override base class guess_type for unity files
        ctype = self.guess_type(path)
        f = open(path, 'rb')
        fs = os.fstat(f.fileno())
        self.send_response(HTTPStatus.OK)
        # check for compression and add header as needed
        if self.path.endswith('.gz'):
            self.send_header('Content-Encoding', 'gzip')
        # the rest is like in the base class implementation
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

    def guess_type(self, path):
        # ensure correct MIME types for Unity WebGL files
        if str(path).endswith('.data.gz'):
            return 'application/octet-stream'
        if str(path).endswith('.js.gz'):
            return 'application/javascript'
        if str(path).endswith('.wasm.gz'):
            return 'application/wasm'
        return super().guess_type(path)
