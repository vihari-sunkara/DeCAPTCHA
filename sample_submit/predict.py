import numpy as np
import cv2,keras
from keras.models import Model, load_model
from sklearn import preprocessing

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.

def decaptcha( filenames ):
    # The use of a model file is just for sake of illustration
    keras.backend.set_learning_phase(0)
    numChars=[]
    codes=[]
    model = load_model('model-tgs-salt-1.h5')
    for filepath in filenames:
        image = cv2.imread(filepath)
        hsv= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower_p = np.array([0,0,150])
        upper_p =np.array([255,255,255])
        mask=cv2.inRange(hsv, lower_p, upper_p)
        thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        numChars.append(len(contours))
        letter_image_regions = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            letter_image_regions.append((x, y, w, h))
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
        predictions = []
        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            letter_image = mask[y - 2:y + h + 2, x - 2:x + w + 2]
            letter_image = cv2.resize(letter_image, (30,30))
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)
            pred = model.predict(letter_image)
            alpha=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
            lb = preprocessing.LabelBinarizer().fit(alpha)
            letter = lb.inverse_transform(pred)[0]
            predictions.append(letter)
        codes.append("".join(predictions))
    return (numChars,codes)
