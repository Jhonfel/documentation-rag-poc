from flask import Flask, send_from_directory

app = Flask(__name__)

# serve index.html
@app.route('/',  methods=["GET",'POST'])
def serve_index():
    return send_from_directory('rag-frontend/dist', 'index.html')

# serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('rag-frontend/dist', path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
