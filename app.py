from flask import Flask, jsonify, abort, send_file, send_from_directory, request
from isi_lib import utils
import cv2
import numpy as np

app = Flask(__name__, static_folder='html')

images = [
    {
        'id': 1,
        'name': u'2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001'
    },
    {
        'id': 2,
        'name': u'2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_002'
    },
    {
        'id': 3,
        'name': u'2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_003'
    },
    {
        'id': 4,
        'name': u'2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_004'
    }
]


@app.route('/', methods=['GET'])
def root():
    return app.send_static_file('index.html')


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/ng-app/<path:path>')
def send_ng(path):
    return send_from_directory('ng-app', path)


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


@app.route('/api/images/', methods=['GET'])
def get_image_list():
    return jsonify(images)


@app.route('/api/images/<int:img_id>', methods=['GET'])
def get_image(img_id):
    img_format = request.args.get('type')

    img = [img for img in images if img['id'] == img_id]
    if len(img) == 0:
        abort(404)
    else:
        img = img[0]

    if img_format == 'tif':
        img_file = '.'.join([img['name'], 'tif'])
        img_path = '/'.join(['data', 'orig-imgs', img_file])
        mime_type = 'image/tiff'
    elif img_format == 'jpg':
        img_file = '.'.join([img['name'], 'jpg'])
        img_path = '/'.join(['data', 'jpeg-imgs', img_file])
        mime_type = 'image/jpeg'
    else:
        abort(404)

    return send_file(img_path, mimetype=mime_type)


@app.route('/api/images/<int:img_id>/id_region', methods=['GET'])
def get_region_id(img_id):
    x = int(request.args.get('x'))
    y = int(request.args.get('y'))
    w = int(request.args.get('w'))
    h = int(request.args.get('h'))

    img = [img for img in images if img['id'] == img_id]
    if len(img) == 0:
        abort(404)
    else:
        img = img[0]

    img_file = '.'.join([img['name'], 'tif'])
    img_path = '/'.join(['data', 'orig-imgs', img_file])

    trained_model, class_map = utils.get_trained_model(
        img_file,
        classifier='svc',
        cached=True
    )

    cv_img = cv2.imread(img_path)
    hsv_image = cv2.cvtColor(np.array(cv_img), cv2.COLOR_BGR2HSV)

    input_dict = {
        'region': hsv_image[y:y+h, x:x+w, :],
        'model': trained_model,
        'class_map': class_map
    }

    # identify best class for region
    predicted_class, probabilities = utils.predict(input_dict)

    if probabilities is not None:
        prob_str = "\n".join(
            ["%s%9.2f%%" % (p[0], p[1] * 100.0) for p in probabilities]
        )
    else:
        prob_str = ''

    return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
