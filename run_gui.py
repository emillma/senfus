from multiprocessing import Process

# import subprocess
from gui import quart_server, flask_app, dash_app
from gui import idp

if __name__ == "__main__":
    # socket_server_process = Process(target=socket_server.run)
    # socket_server_process.start()
    dash_app.run()
    idp.generate_code()
