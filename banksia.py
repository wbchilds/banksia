import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import shutil
import requests
import random


#Takes a gis feaature and finds a random streetview image inside that feature
##TODO: Put the fine saving/bad counts into another place
##TODO: Build in handling for API key elsewhere
##TODO: Should return an image
##TODO: Should also take an array with pitch/fov params etc
def getRandStreetView(feature, filename, bad_counts,apikey):
    rand_lat = random.uniform(feature['properties']['left'], feature['properties']['right'])
    rand_lng = random.uniform(feature['properties']['bottom'], feature['properties']['top'])

    params = "size=600x300&location=" + str(rand_lng) + "," + str(
        rand_lat) + "&pitch=15&heading=151.78&pitch=-0.76&&fov=120&key="+apikey
    metaurl = "https://maps.googleapis.com/maps/api/streetview/metadata?" + params
    url = "https://maps.googleapis.com/maps/api/streetview?" + params

    meta = requests.get(metaurl)
    meta = meta.json()
    print(bad_counts)
    if meta['status'] == "OK":
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
    else:
        bad_counts += 1
        if bad_counts > 5:
            return None
        else:
            getRandStreetView(feature, filename, bad_counts)

##Takes two colour images as ndarrays and concatinates them side by side, returns image
def concatTwoImages(imga, imgb):

    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.uint8)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img

##Takes N colour images from path list and concatinates them side by side, returning image
def concatImages(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:, :, :3]
        if i == 0:
            output = img
        else:
            output = concatTwoImages(output, img)
    return output

##Takes a feature from a geojson and returns X stitched-together images for that feature
##TODO: Should return an image
##TODO: Should take argument int(number of images)
def getCanvasedImages(feature):
    imgs_per_feature = 5
    image_arr = []
    count = 0
    while count <= imgs_per_feature:
        file = "images/london/city/0005/" + str(feature['properties']['id']) + "_" + str(count) + ".jpeg"
        image = getRandStreetView(feature, file, 0)
        #         cv2.imwrite(file,image)
        count += 1

#Changes colours stored as type list and converts to strings readable by qgis
def stringifyColours(data):
            for feature in data['features']:
                            feature['properties']['colour'] = str(feature['properties']['colour'][0])+','+str(feature['properties']['colour'][1])+','+str(feature['properties']['colour'][2])

#Takes a colour and returns an int which is a score of how colourful the image is
#TODO: Take the cubic out of here and put it in a more appropritate place
def howInteresting(colour):
    interestingness = abs(int(colour[0])-int(colour[1]))+abs(int(colour[0])-int(colour[2]))+abs(int(colour[1])-int(colour[2]))
    interestingness = np.power(interestingness,3)
    return interestingness

#Takes a colour, returns True if the colour is too dark and false otherwise
def isDark(colour):
    if colour[0] < 40 and colour [1] < 40 and colour[2] < 40:
        return True
    else:
        return False

#Takes a colour and returns its overall lightness score (the sum of all it's rgb channels)
def getLightness(colour):
    lightness = colour[0]+colour[1]+colour[2]
    return lightness

#Takes an image a returns the average of all it's colours
def getAvgColour(image):
    average_colour = [int(round(image[:, :, i].mean())) for i in range(image.shape[-1])]
    return average_colour

#Takes an image and returns the dominant colour. Dominance is a combination of frequency, lightness and interestingness
#TODO Tidy
#TODO: Implement cubic here
#TODO: Should take parameters for kmeans incl number of colours and implement defaults
#TODO: Should take parameters for dominance and implement defaults
def getDomColour(image):
    # use k-means clustering to create palette with the n_colours=10 most representative colours of the image
    arr = np.float32(image)
    pixels = arr.reshape((-1, 3))

    n_colours = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colours, None, criteria, 10, flags)

    palette = np.uint16(centroids)

    #     quantized = palette[labels.flatten()]
    #     quantized = quantized.reshape(image.shape)

    # the dominant colour is the palette colour which occurs most frequently on the quantised image:

    index_domcol = np.argmax(itemfreq(labels)[:, -1])
    domcol = palette[index_domcol]
    freq_domcol = itemfreq(labels)[:, 1][index_domcol]
    interestingness_domcol = howInteresting(domcol)
    lightness_domcol = getLightness(domcol)
    dominance_domcol = interestingness_domcol * freq_domcol * lightness_domcol
    index = 0
    for colour in palette:
        interestingness = howInteresting(colour)
        freq = itemfreq(labels)[:, 1][index]
        lightness = getLightness(colour)
        dominance = interestingness * freq * lightness
        if dominance > dominance_domcol is not (isDark(colour)):
            domcol = palette[index]
    return domcol

#Takes a number and a range and maps it to a new range maintaining ratios between numbers
def remap( x, oMin, oMax, nMin, nMax ):

    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if nMin == nMax:
        print("Warning: Zero output range")
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result

# Takes a json of features and makes the colours more interesting
#TODO: Lots of fixing up -- should also take parameters for how much it stretches the colours
#TODO: Should actually return a json
def makeInteresting(somedata):
    upper_bound = 255
    lower_bound = 254

    for feature in somedata['features']:
        max_index = feature['properties']['colour'].index(max(feature['properties']['colour']))
        min_index = feature['properties']['colour'].index(min(feature['properties']['colour']))

        feature['properties']['colour'][min_index] = int(round(feature['properties']['colour'][min_index] * 0.8))
        feature['properties']['colour'][max_index] = int(round(feature['properties']['colour'][max_index] * 1.2))

        if feature['properties']['colour'][max_index] > upper_bound:
            upper_bound = feature['properties']['colour'][max_index]
        if feature['properties']['colour'][min_index] < lower_bound:
            lower_bound = feature['properties']['colour'][min_index]

    return [lower_bound, upper_bound]

#Takes a json and compresses the colour space into new bounds
def compressColours(somedata, lower_bound, upper_bound):
    for feature in somedata['features']:
        index = 0
        while index < 3:
            feature['properties']['colour'][index] = int(
                remap(int(feature['properties']['colour'][index]), lower_bound, upper_bound, 0, 255))
            print(feature['properties']['colour'][index])
            index += 1


#TODO: Americanize
#TODO: Should have methods for loading json and saving to files etc
#TODO: Should have method for getting all the images for a given geojson
#TODO: Error handling
#TODO: Parellelize
