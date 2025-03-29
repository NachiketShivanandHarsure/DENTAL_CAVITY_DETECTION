from flask import Flask, render_template, url_for, request


from tensorflow.keras.models import load_model
import pickle
import shutil
from tensorflow.keras.preprocessing import image
import sqlite3
import cv2
import shutil
import cv2

import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
from sklearn.metrics import accuracy_score






# Load the trained model
model = load_model('ResNet50_model.h5')

# Load class names from the pickle file
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

def predict_image(image):
    img =load_img(image, target_size=(150, 150))
    img_array =img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    print(predicted_class_index)
    predicted_class = class_names[predicted_class_index]
    print("predicted_class:",predicted_class)
    prediction1 = prediction.tolist()
    print(prediction1[0][predicted_class_index]*100)
    return predicted_class, prediction1[0][predicted_class_index]*100

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result)==0:
            return render_template('index.html',msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('home.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')



@app.route('/userlog.html', methods=['GET'])
def indexBt():
      return render_template('userlog.html')

@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/Accu_plt.png',
              
              'http://127.0.0.1:5000/static/loss_plt.png',
              'http://127.0.0.1:5000/static/f1_graph.jpg',
              'http://127.0.0.1:5000/static/confusion_matrix.jpg']
    content=['Accuracy Graph',
            'Loss Graph',
            'F1-Score Graph',
            'Confusion Matrix Graph']

            
    
        
    return render_template('graph.html',images=images,content=content)
    



@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        # #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        # #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        # #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                    [-1,-1,-1]])

        # # apply the sharpening kernel to the image
        sharpened =cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)

        predicted_class, accuracy = predict_image("test/"+fileName)
        print("Predicted class:", predicted_class)
        print("Accuracy is:", accuracy)
       
        f = open('acc.txt', 'w')
        f.write(str(accuracy))
        f.close()

        
       
        str_label=""
        accuracy=""
        tre=""
        tre1=""
        rec=""
        Rec1=""
        fl=""
        Fl1=""
        detection=""
        spread=""
        if predicted_class =="mild cavity":
            str_label="mild cavity"
            tre="The Medical Treatment"
            tre1=["Application of fluoride treatments to remineralize teeth."]
            rec="The Dental Cavity Recommendation"
            Rec1=["Avoid sugary snacks and acidic drinks."]
            fl="The Dental Cavity followup"
            Fl1=["Monitor the cavity's progression and adjust fluoride application frequency."]
            spread="There is a  no spread of cavity in the neighbouring teeth"



        elif predicted_class =="moderate cavity":
            str_label="moderate cavity"
            tre="The Medical Treatment"
            tre1=["Dental fillings (composite resin or amalgam)."]
            rec="The Dental Cavity Recommendation"
            Rec1=["Include calcium-rich foods (like milk, cheese, and leafy greens) in your diet."]
            fl="The Dental Cavity followup"
            Fl1=["Replace worn fillings as needed and check for cavity recurrence."]
            spread="There is a  chances that the  cavity may spread to the neighbouring teeth"
                        


        elif predicted_class =="no cavity":
            str_label="no cavity"
            tre="The Medical Treatment"
            tre1=["Routine oral hygiene."]
            rec="The Dental Cavity Recommendation"
            Rec1=["Brush twice a day, use fluoride toothpaste, and floss daily."]
            fl="The Dental Cavity followup"
            Fl1=["Regular dental check-ups every 6 months."]
            
                    
                                    
            

        elif predicted_class =="severe  cavity":
            str_label="severe  cavity"
            tre="The Medical Treatment"
            tre1=["Root canal therapy if the pulp is infected.\n Tooth extraction if damage is irreparable."]
            rec="The Dental Cavity Recommendation"
            Rec1=["Include vitamin D-rich foods (like eggs and fish) to strengthen teeth."]
            fl="The Dental Cavity followup"
            Fl1=["Post-treatment dental implants or dentures if necessary"]
            spread="There is a spread of cavity in the neighbouring teeth"

 
        # Generate the testing graph for accuracy comparison
        A=float( predicted_class =="mild cavity")
        B=float(predicted_class =="moderate cavity")
        C=float(predicted_class =="no cavity")
        D=float(predicted_class =="severe  cavity")

        dic={'Mild':A,'Modarate':B,'Normal':C,'Severe':D}
        algm = list(dic.keys()) 
        accu = list(dic.values()) 
        fig = plt.figure(figsize = (5, 5))  
        plt.bar(algm, accu, color ='maroon', width = 0.3)  
        plt.xlabel("Comparision") 
        plt.ylabel("Accuracy Level") 
        plt.title("Accuracy Comparision between \n Dental cavity detection")
        plt.savefig('static/matrix.png')
                    

       

       
        f = open('acc.txt', 'r')
        accuracy = f.read()
        f.close()
        print(accuracy)

       

        
        
        
        return render_template('results.html', status=str_label,status2=f'accuracy is {accuracy}',Treatment=tre,Treatment1=tre1,Recommendation=rec,Recommendation1=Rec1,FollowUp=fl,FollowUp1=Fl1,detection=detection,spread=spread,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg",ImageDisplay5="http://127.0.0.1:5000/static/matrix.png")
    return render_template('userlog.html')




@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
