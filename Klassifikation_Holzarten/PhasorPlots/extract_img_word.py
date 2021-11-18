import docx2txt
import os
import glob


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

text = ""
#change to needed directory
dir = "/Users/Maxi/signifikanzanalyse-von-fluoreszenzabklingzeiten/Klassifikation_Holzarten/PhasorPlots/PhaserPlotsWord/"



for file in files(dir):
    print(file)

    if not os.path.isdir(dir + "images/" + file):
        os.makedirs(dir + "images/" + file)

    result = docx2txt.process(dir + file, dir + "images/" + file)
    text = " ".join((text, result))


labels = text.split('\n\n\n\n\n\n\n')

for label in labels:
    print(label)
    label = label.replace('\n', '')


print(labels)
