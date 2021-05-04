# =============================================
# The backend's main function. Loads the model and starts the web server
# =============================================

from main_model import init_transformer_model
from web_server import start_web_server

if __name__ == '__main__':
    # Initialise transformer
    init_transformer_model()

    # Start web server (blocking)
    start_web_server()

