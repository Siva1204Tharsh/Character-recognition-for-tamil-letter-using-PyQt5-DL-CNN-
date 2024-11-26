from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import  Sequential,load_model
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

data=[] #input data
labels=[] #output data
classes=247 #class names of output 
cur_path=os.getcwd() # to get current path directory

classs = {
    0: "A-அ",
    1: "Aa-ஆ",
    2: "i-இ",
    3: "ii-ஈ",
    4: "U-உ",
    5: "Oo-ஊ",
    6: "e-எ",
    7: "ae-ஏ",
    8: "ai-ஐ",
    9: "O-ஒ",
    10: "Oa-ஓ",
    11: "Ow-ஔ",
    12: "k-க",
    13: "nG-ங",
    14: "s(ch)-ச",
    15: "Gn(nj)-ஞ",
    16: "t(d)-ட",
    17: "N-ண",
    18: "tha-த",
    19: "nh-ந",
    20: "p-ப",
    21: "m-ம",
    22: "y-ய",
    23: "r-ர",
    24: "l-ல",
    25: "v-வ",
    26: "zh-ழ",
    27: "L-ள",
    28: "R-ற",
    29: "n-ன",
    30: "Ka-க",
    31: "Kaa-கா",
    32: "Ki-கி",
    33: "Kee-கீ",
    34: "Ku-கு",
    35: "Koo-கூ",
    36: "Ke-கெ",
    37: "Kae-கே",
    38: "Kai-கை",
    39: "Ko-கொ",
    40: "Koa-கோ",
    41: "Kou-கௌ",
    42: "nGa-ங",
    43: "nGaa-ஙா",
    44: "nGi-ஙி",
    45: "nGee-ஙீ",
    46: "nGu-ஙு",
    47: "nGoo-ஙூ",
    48: "nGe-ஙெ",
    49: "nGae-ஙே",
    50: "nGai-ஙை",
    51: "nGo-ஙொ",
    52: "nGoa-ஙோ",
    53: "nGou-ஙௌ",
    54: "sa-ச",
    55: "saa-சா",
    56: "si-சி",
    57: "see-சீ",
    58: "su-சு",
    59: "soo-சூ",
    60: "se-செ",
    61: "sae-சே",
    62: "sai-சை",
    63: "so-சொ",
    64: "soa-சோ",
    65: "sou-சௌ",
    66: "Gna-ஞ",
    67: "Gnaa-ஞா",
    68: "Gni-ஞி",
    69: "Gnee-ஞீ",
    70: "Gnu-ஞு",
    71: "Gnoo-ஞூ",
    72: "Gne-ஞெ",
    73: "Gnae-ஞே",
    74: "Gnai-ஞை",
    75: "Gno-ஞொ",
    76: "Gnoa-ஞோ",
    77: "Gnou-ஞௌ",
    78: "ta-ட",
    79: "taa-டா",
    80: "ti-டி",
    81: "tee-டீ",
    82: "tu-டு",
    83: "too-டூ",
    84: "te-டெ",
    85: "tae-டே",
    86: "tai-டை",
    87: "to-டொ",
    88: "toa-டோ",
    89: "tou-டௌ",
    90: "Na-ண",
    91: "Naa-ணா",
    92: "Ni-ணி",
    93: "Nee-ணீ",
    94: "Nu-ணு",
    95: "Noo-ணூ",
    96: "Ne-ணெ",
    97: "Nae-ணே",
    98: "Nai-ணை",
    99: "No-ணொ",
    100: "Noa-ணோ",
    101: "Nou-ணௌ",
    102: "tha-த",
    103: "thaa-தா",
    104: "thi-தி",
    105: "thee-தீ",
    106: "thu-து",
    107: "thoo-தூ",
    108: "the-தெ",
    109: "thae-தே",
    110: "thai-தை",
    111: "tho-தொ",
    112: "thoa-தோ",
    113: "thou-தௌ",
    114: "nha-ந",
    115: "nhaa-நா",
    116: "nhi-நி",
    117: "nhee-நீ",
    118: "nhu-நு",
    119: "nhoo-நூ",
    120: "nhe-நெ",
    121: "nhae-நே",
    122: "nhai-நை",
    123: "nho-நொ",
    124: "nhoa-நோ",
    125: "nhou-நௌ",
    126: "pa-ப",
    127: "paa-பா",
    128: "pi-பி",
    129: "pee-பீ",
    130: "pu-பு",
    131: "poo-பூ",
    132: "pe-பெ",
    133: "pae-பே",
    134: "pai-பை",
    135: "po-பொ",
    136: "poa-போ",
    137: "pou-பௌ",
    138: "ma-ம",
    139: "maa-மா",
    140: "mi-மி",
    141: "mee-மீ",
    142: "mu-மு",
    143: "moo-மூ",
    144: "me-மெ",
    145: "mae-மே",
    146: "mai-மை",
    147: "mo-மொ",
    148: "moa-மோ",
    149: "mou-மௌ",
    150: "ya-ய",
    151: "yaa-யா",
    152: "yi-யி",
    153: "yee-யீ",
    154: "yu-யு",
    155: "yoo-யூ",
    156: "ye-யெ",
    157: "yae-யே",
    158: "yai-யை",
    159: "yo-யொ",
    160: "yoa-யோ",
    161: "you-யௌ",
    162: "ra-ர",
    163: "raa-ரா",
    164: "ri-ரி",
    165: "ree-ரீ",
    166: "ru-ரு",
    167: "roo-ரூ",
    168: "re-ரெ",
    169: "rae-ரே",
    170: "rai-ரை",
    171: "ro-ரொ",
    172: "roa-ரோ",
    173: "rou-ரௌ",
    174: "la-ல",
    175: "laa-லா",
    176: "li-லி",
    177: "lee-லீ",
    178: "lu-லு",
    179: "loo-லூ",
    180: "le-லெ",
    181: "lae-லே",
    182: "lai-லை",
    183: "lo-லொ",
    184: "loa-லோ",
    185: "lou-லௌ",
    186: "va-வ",
    187: "vaa-வா",
    188: "vi-வி",
    189: "vee-வீ",
    190: "vu-வு",
    191: "voo-வூ",
    192: "ve-வெ",
    193: "vae-வே",
    194: "vai-வை",
    195: "vo-வொ",
    196: "voa-வோ",
    197: "vou-வௌ",
    198: "zha-ழ",
    199: "zhaa-ழா",
    200: "zhi-ழி",
    201: "zhee-ழீ",
    202: "zhu-ழு",
    203: "zhoo-ழூ",
    204: "zhe-ழெ",
    205: "zhae-ழே",
    206: "zhai-ழை",
    207: "zho-ழொ",
    208: "zhoa-ழோ",
    209: "zhou-ழௌ",
    210: "La-ள",
    211: "Laa-ளா",
    212: "Li-ளி",
    213: "Lee-ளீ",
    214: "Lu-ளு",
    215: "Loo-ளூ",
    216: "Le-ளெ",
    217: "Lae-ளே",
    218: "Lai-ளை",
    219: "Lo-ளொ",
    220: "Loa-ளோ",
    221: "Lou-ளௌ",
    222: "Ra-ற",
    223: "Raa-றா",
    224: "Ri-றி",
    225: "Ree-றீ",
    226: "Ru-று",
    227: "Roo-றூ",
    228: "Re-றெ",
    229: "Rae-றே",
    230: "Rai-றை",
    231: "Ro-றொ",
    232: "Roa-றோ",
    233: "Rou-றௌ",
    234: "na-ன",
    235: "naa-னா",
    236: "ni-னி",
    237: "nee-னீ",
    238: "nu-னு",
    239: "noo-னூ",
    240: "ne-னெ",
    241: "nae-னே",
    242: "nai-னை",
    243: "no-னொ",
    244: "noa-னோ",
    245: "nou-னௌ",
    246: "ak-ஃ"
}
 

#Retrieving the images and their labels
print("Obtaining Images & its Labels..............")
for i in range(classes):
    path=os.path.join(cur_path,'Dataset/Train/',str(i))
    images=os.listdir(path)

    for a in images:
        try:
            image=Image.open(path+'\\'+a)
            image=image.resize((64,64))
            image=np.array(image)
            data.append(image)
            labels.append(i)
            print("{0} Loaded".format(a))
        except:
            print("Error loading image")
print("Dataset Loaded")

#Converting lists into numpy arrays
data=np.array(data)
labels=np.array(labels)

##print(data.shape, labels.shape)

#Splitting training and testing dataset
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=42)

##print(X_train.shape, X_test.shape, y+_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train=to_categorical(y_train,247) 
y_test=to_categorical(y_test,247)

#Class - 247
#o/p 2 - [0,0,1,0,0,......0]
#o/p 5 - [0,0,0,0,0,1,0,0,0.....]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "           Tamil Letter RECOGNITION"))
        self.Classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
        if fileName: # If the user gives a file
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.imageLbl.setPixmap(pixmap) # Set the pixmap onto the label
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center

    def classifyFunction(self):
        model = load_model('model.h5')
        print("Loaded model from disk")
        path2=self.file
        print(path2)
        test_image = Image.open(path2)
        test_image = test_image.resize((64,64))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.array(test_image)

        result = model.predict(test_image)[0]
        predicted_class_index = result.argmax() # Get the index of the predicted class with the highest probability
        sign = classs[predicted_class_index]
        print(sign)
        self.textEdit.setText(sign)

    def trainingFunction(self):
        self.textEdit.setText("Training under process...")
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(247, activation='softmax'))
        print("Initialized model")

        # Compilation of the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),steps_per_epoch=len(X_train)//32)
        model.save("model.h5") #save the model

        ## evalute the model
        score = model.evaluate(X_test, y_test, verbose=0)
        # print('Test loss:', score[0])
        print(f"Test loss: {score[0]:.2f}, Test accuracy: {score[1]*100:.2f}%")

        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('Accuracy1.png')

        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('Loss1.png')
        self.textEdit.setText("Saved Model & Graph to disk")
        
        
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()            
    sys.exit(app.exec_())           
