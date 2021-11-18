import docx2txt
import os
import glob
import fnmatch


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

text = ""
#change to needed directory
dir = "C:/Users/Maxi/signifikanzanalyse-von-fluoreszenzabklingzeiten/Klassifikation_Holzarten/PhasorPlots/PhaserPlotsWord/"



for file in files(dir):
    print(file)

    if not os.path.isdir(dir + "images/" + file):
        os.makedirs(dir + "images/" + file)

    result = docx2txt.process(dir + file, dir + "images/" + file)
    text = "\n".join((text, result))



#text = text.replace("\n", "#")

labels = text.split('\n')
labels = fnmatch.filter(labels, '*M*')


