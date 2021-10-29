from Klassifikation_Holzarten.Auswertung_nd2.collect_PLT_MLT import collect_plt_mlt, collect_files_for_each_type
from Klassifikation_Holzarten.Auswertung_nd2.helper_functions import *
from Klassifikation_Holzarten.Auswertung_nd2.read_nd2 import readnd2File_ideal
from Klassifikation_Holzarten.Auswertung_nd2.prepare_data_mlp import create_dataset_from_nd2
import tifffile as tiff
from PIL import Image


"""Please make sure when using the script to set your path wright"""

"""Set values to True if you want to execute the specific parts of the script"""
readnd2_ideal = True
create_mlp_dataset = True
create_cnn_images = True

"""Read nd2 files and create TIF images"""

images = []
exposure_time = []
if readnd2_ideal:
    images, exposure_time = readnd2File_ideal(r'F:\nd2_files\405 nm x10 M16 100pct Idealholz test')


# all_PLT_MLT_multi = {"PLT": {}, "MLT": {}}
# all_PLT_MLT_single = {"PLT": {}, "MLT": {}}

""" Collect only Phase Lifetime and Modulation Lifetime TIF and create PNG Images"""


def save_images(files, save_path):
    for type in files:
        for tif in type:
            im = tiff.imread(tif)
            name = Image.open(tif)
            name = name.filename.split('.')[0]
            name = name.split('\\')[6]
            art = ""
            if "Fichte" in name:
                art = "Fichte"

            elif "Ahorn" in name:
                art = "Ahorn"
            elif "Buche" in name:
                art = "Buche"
            elif "Kiefer" in name:
                art = "Kiefer"
            elif "Laerche" in name:
                art = "Laerche"
            elif "Eiche" in name:
                art = "Eiche"

            if "488" in name:
                plt.imsave(save_path + art + '/' + name + '.png', im)
                print(name)
            elif "445" in name:
                plt.imsave(save_path + art + '/' + name + '.png', im)
                print(name)
            elif "405" in name:
                plt.imsave(save_path + art + '/' + name + '.png', im)
                print(name)


if create_cnn_images:
    files = list(dict.fromkeys(collect_plt_mlt()))
    files = sorted(files)

    ideal_files = collect_files_for_each_type(files)

    save_images(ideal_files, '../Neuronale_netze/images/405nm_x10_100pct_test/')


""" Prepare Data for MLP"""
if create_mlp_dataset:
    create_dataset_from_nd2(r'F:\nd2_files\405 nm x10 M16 100pct Idealholz test')