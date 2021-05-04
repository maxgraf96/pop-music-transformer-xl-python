# =============================================
# The web server used for communication between this backend and the Node.JS server / website in browser
# =============================================

import http.server
import socketserver
import sys

import requests

from main_model import generate_from_scratch, generate_from_conditioned, generate_from_recorded

# TODO change this in frontend (app.js) and here if port 5000 is taken by some other process on your system!!
FRONTEND_PORT = 5000


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom http handler to communicate between Node.JS and python
    """
    def do_GET(self):
        if self.path == '/':
            pass
        # Called when user presses "Generate" button
        elif self.path == '/from_scratch':
            print("Received request, generating from scratch...")
            # Generate notes and chords
            generate_from_scratch()
            # Notify Node.JS
            send_midi_ready()

        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self, urlparse=None):
        # Generate from conditioned
        if self.path == "/from_conditioned":
            print("Received request, generating from conditioned...")

            # Extract player index from request
            content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
            post_data = self.rfile.read(content_length)  # <--- Gets the data itself
            data = post_data.decode('utf-8')
            player_idx = int(data.replace("\"", ""))
            # Generate notes and chords
            generate_from_conditioned(player_idx)
            send_midi_ready()
        # Generate from recorded
        if self.path == "/from_recorded":
            print("Received request, generating from recorded...")
            try:
                generate_from_recorded()
                send_midi_ready()
            except Exception:
                print()
                print("xxx")
                print("Got empty MIDI sequence. Can't generate from empty sequence...")
                print("Please make sure that your recorded sequence contains at least one MIDI note.")
                print("xxx")
                print()


def send_midi_ready():
    """
    Helper function to notify frontend that MIDI is ready
    :return: None
    """
    data = {}
    r = requests.post('http://127.0.0.1:' + str(FRONTEND_PORT) + '/midi_ready', data=data)
    print(r.content)


def send_port(port):
    """
    Helper function to notify frontend about python server port
    :param port: The port
    :return: None
    """
    data = {
        'new_port': str(port)
    }
    # If this request goes through we are successfully connected to the frontend
    r = requests.post('http://127.0.0.1:' + str(FRONTEND_PORT) + '/python_port', data=data)
    # print(r.content)
    print()
    print("Successfully connected to the frontend!")


def start_web_server():
    """
    Start the web server
    :return: None
    """
    # Create an object of the above class
    handler_object = MyHttpRequestHandler

    PORT = 12000
    try:
        my_server = socketserver.TCPServer(("localhost", PORT), handler_object)
    except:
        print("Port ", PORT, " already in use, trying ", PORT + 1)
        PORT = PORT + 1
        my_server = socketserver.TCPServer(("localhost", PORT), handler_object)

    try:
        send_port(PORT)
    except Exception as e:
        print()
        print("xxxxxxxxxxxxxxxxxxxxx")
        print("Frontend not running. Please start the Node.JS server first, then run main.py again.")
        print("Refer to the submission README file for instructions on the execution order.")
        print("Exiting...")
        print("xxxxxxxxxxxxxxxxxxxxx")
        print()
        sys.exit()

    # Start the server
    print("Serving python server at :" + str(PORT))
    print("Please navigate back to the frontend.")
    print("Shortcut: Click http://localhost:" + str(FRONTEND_PORT) + '/')
    print()

    # Serve continuously
    try:
        my_server.serve_forever()
    # shutdown gracefully
    except Exception:
        print("Shutting down")
        my_server.shutdown()
