import tempfile
import os

from flask import Flask, request
from data_url import DataURL, construct_data_url
from mustacher.mustacher import mustache_image_data

app = Flask(__name__)

@app.route("/mustachify", methods=['POST'])
def mustachify_image():
    url = DataURL.from_url(request.get_data(as_text=True))

    mustached_image = mustache_image_data(url.data)

    mustache_url = construct_data_url(mime_type="image/jpeg", base64_encode=True, data=mustached_image)
    with open('out.jpg', 'wb') as f:
        f.write(mustached_image)
        f.flush

    return mustache_url

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response

if __name__ == "__main__":
    app.run()
