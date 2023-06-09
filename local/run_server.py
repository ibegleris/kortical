import os
import alex.main
from kortical.app import get_config

app_config = get_config(format='yaml')
api_key = app_config['api_key']

if __name__ == '__main__':
    port = 5001
    print(f"Paste the following into your browser to access the app:\n\n[http://127.0.0.1:{port}?api_key={api_key}]")
    os.environ['SERVER_RUNNING_LOCALLY'] = "TRUE"
    alex.main.app.run(debug=True, port=port)