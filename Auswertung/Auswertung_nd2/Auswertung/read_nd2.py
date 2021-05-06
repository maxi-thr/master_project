import os
from nd2reader import ND2Reader
import tifffile
import glob
import pandas as pd
import javabridge
import bioformats


def readnd2File():
    my_path = 'nd2_files'
    files = glob.glob(my_path + '/**/*.nd2', recursive=True)
    javabridge.start_vm(class_path=bioformats.JARS)
    for file in files:

        with bioformats.ImageReader(file) as reader:
            values = reader.read()

        """ Read all ND2 Files and create a new path """
        with ND2Reader(file) as images:
            name_data = os.path.basename(file)
            name_data_struct = name_data.split('.')[0]
            new_path = file.replace(name_data, '') + '\\' + name_data_struct
            filepath = os.path.join(new_path, name_data_struct)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                os.makedirs(new_path + '\\' + "excel")
                os.makedirs(new_path + '\\' + "tif")
                print("new path created: " + new_path)
            channels = images.metadata['channels']
            i = 0
            for channel in channels:
                matrix = values[:, :, i]
                df = pd.DataFrame(matrix)
                if not os.path.isfile(new_path + '\\' + "tif" + name_data_struct + "_" + channel + ".tif"):
                    tifffile.imsave(
                        os.path.join(new_path + '\\' + "tif", name_data_struct + "_" + channel + ".tif"), matrix)
                    #df.to_excel(excel_writer=os.path.join(new_path + '\\' + "excel", name_data_struct + "_" + channel + ".xlsx"))
                i = i + 1
                print("tif file created: " + name_data_struct + "_" + channel + ".tif")


def readnd2File_ideal():
    my_path = 'nd2_files/Ideal'
    files = glob.glob(my_path + '/**/*.nd2', recursive=True)
    javabridge.start_vm(class_path=bioformats.JARS)
    for file in files:

        with bioformats.ImageReader(file) as reader:
            values = reader.read()

        """ Read all ND2 Files and create a new path """
        with ND2Reader(file) as images:
            name_data = os.path.basename(file)
            name_data_struct = name_data.split('.')[0]
            new_path = file.replace(name_data, '') + '\\' + name_data_struct
            filepath = os.path.join(new_path, name_data_struct)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                os.makedirs(new_path + '\\' + "excel")
                os.makedirs(new_path + '\\' + "tif")
                print("new path created: " + new_path)
            channels = images.metadata['channels']
            i = 0
            for channel in channels:
                matrix = values[:, :, i]
                df = pd.DataFrame(matrix)
                if not os.path.isfile(new_path + '\\' + "tif" + name_data_struct + "_" + channel + ".tif"):
                    tifffile.imsave(
                        os.path.join(new_path + '\\' + "tif", name_data_struct + "_" + channel + ".tif"), matrix)
                    #df.to_excel(excel_writer=os.path.join(new_path + '\\' + "excel", name_data_struct + "_" + channel + ".xlsx"))
                i = i + 1
                print("tif file created: " + name_data_struct + "_" + channel + ".tif")


