import os, sys
from flask import Flask,  Markup, escape, request, Response, g, make_response, redirect, url_for
from flask.templating import render_template
from werkzeug.utils import secure_filename
# from image_test import Test
import detection
import threading
from flask import jsonify, request
from flask import Flask, render_template
from flask_socketio import SocketIO

secret = os.urandom(24).hex()

app = Flask(__name__)
app.logger.info("Starting...")
app.config['SECRET_KEY'] = secret
app.logger.critical("secret: %s" % secret)
socketio = SocketIO(app)

# Main page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index')
def return_index():
    return render_template('index.html')


@app.route('/det_get')
def nst_get():
    return render_template('det_get.html')


@app.route('/det_get2')
def nst_get2():
    return render_template('det_get2.html')


@app.route('/det_post', methods=['GET', 'POST'])
def nst_post():
    if request.method == 'POST':
        # User Image (target image)
        user_img = request.files['user_img']
        user_img.save('./static/images/' + str(user_img.filename))
        user_img_path = './static/images/' + str(user_img.filename)

        det = detection.Detector()

        #for radio button
        option = request.form['options']
        if option == "tree":
            ###for tree test
            user_result = det.test_tree(user_img_path)
            user_result_path = str(user_result)

            return render_template('det_post.html', user_img=user_img_path, user_result=user_result_path, user_url = "")

        elif option == "cat":
            ###for cat test
            user_result = det.test_cat(user_img_path)
            user_result_path = str(user_result)
            #url = os.path.join(os.path.join('./static', 'images'), 'new_plot.png')
            url_path = './static/images/new_plot.jpg'
            return render_template('det_post.html', user_img = user_img_path, user_result = user_result_path, user_url = url_path)

        ###for cat test
        #user_result = det.test_cat(user_img_path)
        #user_result_path = str(user_result)

    #return render_template('det_post.html', user_img=user_img_path, user_result=user_result_path)

if __name__ == "__main__":
    """
	# get port. Default to 8080
	port = int(os.environ.get('PORT', 8000))

	# run app
	app.run(host='0.0.0.0', port=port)
    """
    socketio.run(app, debug=True)
