import docx2txt
import os
import glob


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

text = ""
dir = "/home/linux/signifikanzanalyse-von-fluoreszenzabklingzeiten/Klassifikation_Holzarten/PhasorPlots/PhasorPlotsWord/"



for file in files("../PhasorPlotsWord"):
    print (file)

    if not os.path.isdir(dir + "images/" + file):
        os.makedirs(dir + "images/" + file)

    result = docx2txt.process(dir + file, dir + "images/" + file)
    text = " ".join((text, result))

if os.path.isfile('/home/linux/signifikanzanalyse-von-fluoreszenzabklingzeiten/Klassifikation_Holzarten/PhasorPlots/PhasorPlotsWord/.~lock.Phaser_Plots_04112021.docx#'):
    os.remove('/home/linux/signifikanzanalyse-von-fluoreszenzabklingzeiten/Klassifikation_Holzarten/PhasorPlots/PhasorPlotsWord/.~lock.Phaser_Plots_04112021.docx#')

print(text)