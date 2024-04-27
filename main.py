from flask import Flask,request,render_template,send_from_directory
import os
import joblib
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import time
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'joblib','.h5', 'jpg', 'jpeg'}


def vmres(model_path,input_data):
  model = load_model(model_path)
  input_data = np.array(input_data).reshape(1, -1)
  predictions = model.predict(input_data)
  return predictions

def imres(model_path,image_path): 
  model = load_model(model_path)
  img = image.load_img(image_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0
  prediction = model.predict(img_array)
  class_label = "Parkinson's" if prediction[0][0] > 0.5 else 'Healthy'
  return class_label

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def savemodal(image):
    if image:
      filename = secure_filename(image.filename)
      filename,extension=os.path.splitext(filename)
      filename=f"{str(time.time()).replace('.','_')}{extension}"
      image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      image.save(image_path)
      return filename

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method=='POST':
    form_values = [
            request.form['MDVP_Fo'],
            request.form['MDVP_Fhi'],
            request.form['MDVP_Flo'],
            request.form['MDVP_Jitter'],
            request.form['MDVP_Jitter_Abs'],
            request.form['MDVP_RAP'],
            request.form['MDVP_PPQ'],
            request.form['Jitter_DDP'],
            request.form['MDVP_Shimmer'],
            request.form['MDVP_Shimmer_dB'],
            request.form['Shimmer_APQ3'],
            request.form['Shimmer_APQ5'],
            request.form['MDVP_APQ'],
            request.form['Shimmer_DDA'],
            request.form['NHR'],
            request.form['HNR'],
            request.form['RPDE'],
            request.form['DFA'],
            request.form['spread1'],
            request.form['spread2'],
            request.form['D2'],
            request.form['PPE']
        ]
    
    vm=request.files['voicemodelFile']
    im=request.files['imgmodelFile']
    image=request.files['imageFile']
    vmfname=savemodal(vm)
    imfname=savemodal(im)
    imagefname=savemodal(image)
    print(vmfname,imfname,imagefname)
    ires=imres(f"uploads\{imfname}",f"uploads\{imagefname}")
    form_values=[float(i) for i in form_values ]
    vres=vmres(f"uploads\{vmfname}",form_values)
    print("output printing----")
    print(form_values)
    return render_template('output.html',data=form_values,item=imagefname,ires=ires,vres=vres)
  return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    uploads_folder = os.path.join(app.root_path, 'uploads')
    return send_from_directory(uploads_folder, filename)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80,debug=True)
