"""Starts WebShop Flask server on a free port, returns the port. Used by eval."""
import subprocess, socket, time, sys

def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_webshop_server(timeout=60):
    port = find_free_port()
    proc = subprocess.Popen(
        ['python', '-m', 'web_agent_site.app', '--port', str(port)],
        cwd='third_party/WebShop',
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f'http://localhost:{port}')
            print(f'WebShop server ready on port {port}')
            return proc, port
        except:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError(f'WebShop server failed to start within {timeout}s')

if __name__ == '__main__':
    proc, port = start_webshop_server()
    print(port)
    proc.wait()
