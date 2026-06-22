import asyncio
import json
import socketserver
import websockets

from UI.HTTP_server import UnityHTTPRequestHandler
from threading import Thread

class unity_player():
    """
    A simple interface to a unity WebGL engine for embedding via HTML.Iframe

    Callable interface:
     - __init__(supply_HTTP_server, HTTP_unity_url) 
        + supply_HTTP_server: if set to true the app will start a local HTTP server to serve the unity index.html
        + HTTP_unity_url: location of the unity index.html
    """

    def __init__(self, dash_app, supply_HTTP_server = True, HTTP_unity_url = 'http://localhost:8001/unity/index.html'):
        self.dash_app = dash_app
        
        if supply_HTTP_server:
            self.HTTP_unity_server = socketserver.TCPServer(('', 8001), UnityHTTPRequestHandler)

            self.HTTP_unity_server_thread = Thread(target = self.run_HTTP_unity_server, daemon = True)
            self.HTTP_unity_server_thread.start()

            self.HTTP_unity_url = HTTP_unity_url

        self.unity_data_server_thread = Thread(target = self.run_unity_data_server, daemon = True)
        self.unity_data_server_thread.start()

    def run_HTTP_unity_server(self):
        self.HTTP_unity_server.serve_forever()
    
    def run_unity_data_server(self):
        async def loop():
            async with websockets.serve(self._unity_data_handler, 'localhost', 8002):
                await asyncio.Future()
        asyncio.run(loop())

    async def _unity_data_handler(self, websocket):
        async for message in websocket:
            try:
                # parse the request from Unity
                request = json.loads(message)
                # send data back to Unity
                await websocket.send(json.dumps(self.dash_app._assemble_unity_response(request)))
            except json.JSONDecodeError:
                raise SystemExit('Received invalid JSON message {} from Unity.'.format(message))
