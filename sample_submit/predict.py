import numpy as np

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that 
# were given. The evaluation code may give unexpected results if this convention is not followed.

def decaptcha( filenames ):
    numChars = 3 * np.ones( (len( filenames ),) )
    # The use of a model file is just for sake of illustration
    file = open( "model.txt", "r" )
    codes = file.read().splitlines()
    file.close()
    return (numChars, codes)