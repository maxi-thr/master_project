import docx2txt
import os
import glob
import fnmatch
import shutil
import itertools
import sys

#function to open files in a specied path
def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


text = ""
img_count=0
#change to needed directory
dir = "C:/Users/Maxi/signifikanzanalyse-von-fluoreszenzabklingzeiten/Klassifikation_Holzarten/PhasorPlots/PhaserPlotsWord/"
dir_img = "C:/Users/Maxi/signifikanzanalyse-von-fluoreszenzabklingzeiten/Klassifikation_Holzarten/PhasorPlots/PhaserPlotsWord/images/"

#remove all previously existing images for safety
shutil.rmtree(dir + "images")
os.makedirs(dir + "images")

#loop over all word files and scraping images into specifies folder as well as adding all the text into variable 'text'
for file in files(dir):
    
    os.makedirs(dir + "images/" + file)

    result = docx2txt.process(dir + file, dir + "images/" + file)
    result = result + "\n"
    text = "\n".join((text, result))


#split the text into labels 
labels = text.split('\n')
labels = fnmatch.filter(labels, '*M*')

#compare labels against each other to be sure there is no double 
for a, b in itertools.combinations(labels, 2):
    if a == b:
        print("Problem")
        print(a)
        print(b)
        sys.exit(1)
    
#get all directories in the image folder
imag_dirs = os.listdir(dir_img)
for img_folder in imag_dirs:
    temp_path = dir_img + img_folder
    img_count += len([name for name in os.listdir(temp_path) if os.path.isfile(os.path.join(temp_path, name))])

i=0


#loop over all the images in every subfolder of 'image' and rename them to the according label + move all named images into the Datastore

#adjust destination for own PC!
destination = "C:/Users/Maxi/signifikanzanalyse-von-fluoreszenzabklingzeiten/Klassifikation_Holzarten/PhasorPlots/PhaserPlotsWord/PhasorPlotsDatastorage"
if not os.path.isdir(destination):
    os.makedirs(destination)

for img_folder in imag_dirs:
    temp_path = dir_img + img_folder
    for file in files(temp_path):
        os.rename(temp_path + '/' + file, temp_path + '/' + labels[i] + '.png')
        shutil.copy(temp_path + '/' + labels[i] + '.png', destination)
        i += 1

