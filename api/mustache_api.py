import tempfile
import os

from flask import Flask, request
from data_url import DataURL
from mustacher.mustacher import mustache_image_data

app = Flask(__name__)

@app.route("/mustachify", methods=['POST'])
def mustachify_image():
    url = DataURL.from_url(request.get_data(as_text=True))

    mustached_image = mustache_image_data(url.data)

    with open('out.jpg', 'wb') as f:
        f.write(mustached_image)
        f.flush

    return mustached_image


if __name__ == "__main__":
    app.run()
