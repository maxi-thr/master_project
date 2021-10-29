import os
from nd2reader import ND2Reader
import pims
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
        with pims.ND2_Reader(file) as images:
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


def readnd2File_ideal(path):
    all_images = []
    all_exposure_time = []
    my_path = path
    files = glob.glob(my_path + '/**/*.nd2', recursive=True)
    files = list(dict.fromkeys(files))
    javabridge.start_vm(class_path=bioformats.JARS)
    for file in files:
        """ Read all ND2 Files and create a new path """
        with ND2Reader(file) as images:
            raw = ND2Reader(file).parser._raw_metadata
            exposure_time = raw.camera_exposure_time
            all_images.append(images)
            all_exposure_time.append(exposure_time[0])
            name_data = os.path.basename(file)
            name_data_struct = name_data.split('.')[0]
            new_path = file.replace(name_data, '') + '\\' + name_data_struct
            filepath = os.path.join(new_path, name_data_struct)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                #os.makedirs(new_path + '\\' + "excel")
                os.makedirs(new_path + '\\' + "tif")
                print("new path created: " + new_path)
            channels = images.metadata['channels']
            i = 0
            with bioformats.ImageReader(file) as reader:
                metadata = reader.metadata
                print(metadata)
                values = reader.read()
            for channel in channels:
                matrix = values[:, :, i]
                df = pd.DataFrame(matrix)
                if not os.path.isfile(new_path + '\\' + "tif" + name_data_struct + "_" + channel + "_" + str(exposure_time[0]) + ".tif"):
                    tifffile.imsave(
                        os.path.join(new_path + '\\' + "tif", name_data_struct + "_" + channel + "_" + str(exposure_time[0]) + ".tif"), matrix)
                    #df.to_excel(excel_writer=os.path.join(new_path + '\\' + "excel", name_data_struct + "_" + channel + ".xlsx"))
                i = i + 1
                print("tif file created: " + name_data_struct + "_" + channel + "_" + str(exposure_time[0]) + ".tif")
    return all_images, all_exposure_time


def delete():

    my_path = r'F:\nd2_files\445 nm x10 M16 100pct Idealholz'
    files = glob.glob(my_path + '/**/*Phase Lifetime.tif', recursive=True)
    for file in files:
        os.remove(file)
        print('deleted: ' + file)


def rename_files():
    path = r'D:\nd2_altholz\488\Praepariert\Laerche'
    files = os.listdir(path)
    num_mes = 16
    i = 1
    y = 1
    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join('Laerche_F' + str(y) + '_M' + str(i) + '__Praepariert_488nm_x10_100pct_praepariert.nd2')))
        i = i +1
        if i == num_mes + 1:
            y = y + 1
            i = 1
