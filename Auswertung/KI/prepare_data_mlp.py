import glob
import tifffile as tiff
from PIL import Image
import pandas as pd
import javabridge
import bioformats
from nd2reader import ND2Reader
import timing


def collect_plt_mlt_intensity():

    my_path = r'F:\nd2_files\488 nm x10M16 100pct Idealholz'
    files = glob.glob(my_path + '/**/**/*Lifetime*.tif', recursive=True)
    intensity_files = glob.glob(my_path + '/**/**/*Intensity*.tif', recursive=True)
    files += intensity_files
    files = list(dict.fromkeys(files))
    return files


def create_dataset(files):
    all_data = []
    for file in files:
        data = []
        im = tiff.imread(file)
        name = Image.open(file)
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

        std = im.std()
        mean = im.mean()

        data.append(art)
        data.append(std)
        data.append(mean)
        all_data.append(data)

    columns = ["art", "std", "mean"]
    all_data = pd.DataFrame(data=all_data, columns=columns)
    return all_data


def create_dataset_from_nd2():

    my_path = r'/media/linux/Seagate Expansion Drive/nd2_files'
    files = glob.glob(my_path + '/488*/**/*.nd2', recursive=True)
    files = list(dict.fromkeys(files))
    javabridge.start_vm(class_path=bioformats.JARS)
    all_data = []
    for file in files:
        """ Read all ND2 Files and create a new path """
        with ND2Reader(file) as images:
            data = []
            raw = ND2Reader(file).parser._raw_metadata
            exposure_time = raw.camera_exposure_time
            name_data_struct = file.split('.')[0]
            art = ""
            if "Fichte" in name_data_struct:
                art = "Fichte"
            elif "Ahorn" in name_data_struct:
                art = "Ahorn"
            elif "Buche" in name_data_struct:
                art = "Buche"
            elif "Kiefer" in name_data_struct:
                art = "Kiefer"
            elif "Laerche" in name_data_struct:
                art = "Laerche"
            elif "Eiche" in name_data_struct:
                art = "Eiche"

            data.append(art)
            channels = images.metadata['channels']
            i = 0
            with bioformats.ImageReader(file) as reader:
                values = reader.read()
            for channel in channels:
                if "Intensity" in channel:
                    matrix = values[:, :, i]
                    matrix = matrix * (1000/int(exposure_time[0]))
                    std = matrix.std()
                    mean = matrix.mean()
                    data.append(std)
                    data.append(mean)
                elif "Phase Lifetime" in channel:
                    matrix = values[:, :, i]
                    std = matrix.std()
                    mean = matrix.mean()
                    data.append(std)
                    data.append(mean)
                elif "Modulation Lifetime" in channel:
                    matrix = values[:, :, i]
                    std = matrix.std()
                    mean = matrix.mean()
                    data.append(std)
                    data.append(mean)
                i = i + 1

        all_data.append(data)

    columns = ["art", "intensity_std", "intensity_mean", "phase_std", "phase_mean", "mod_std", "mod_mean"]
    all_data = pd.DataFrame(data=all_data, columns=columns)
    save_path = "csv/"
    if "488" in files[0]:
        all_data.to_csv(save_path + "dataset_488.csv")
    elif "445" in files[0]:
        all_data.to_csv(save_path + "dataset_445.csv")
    elif "405" in files[0]:
        all_data.to_csv("dataset_405.csv")

    javabridge.kill_vm()


def create_dataset_from_nd2_altholz():

    my_path = r'D:\nd2_altholz\405'
    files = glob.glob(my_path + '/**/**/*.nd2', recursive=True)
    files = list(dict.fromkeys(files))
    javabridge.start_vm(class_path=bioformats.JARS)
    all_data = []
    for file in files:
        """ Read all ND2 Files and create a new path """
        with ND2Reader(file) as images:
            data = []
            raw = ND2Reader(file).parser._raw_metadata
            exposure_time = raw.camera_exposure_time
            name_data_struct = file.split('.')[0]
            classification = ""
            # if "Fichte" in name_data_struct:
            #     art = "Fichte"
            # elif "Ahorn" in name_data_struct:
            #     art = "Ahorn"
            # elif "Buche" in name_data_struct:
            #     art = "Buche"
            # elif "Kiefer" in name_data_struct:
            #     art = "Kiefer"
            # elif "Laerche" in name_data_struct:
            #     art = "Laerche"
            # elif "Eiche" in name_data_struct:
            #     art = "Eiche"

            if "AI\\" in name_data_struct:
                classification = "AI"
            elif "AII\\" in name_data_struct:
                classification = "AII"
            elif "AIII\\" in name_data_struct:
                classification = "AIII"
            elif "AIV\\" in name_data_struct:
                classification = "AIV"
            else:
                classification = "Praep"

            data.append(classification)
            channels = images.metadata['channels']
            i = 0
            with bioformats.ImageReader(file) as reader:
                values = reader.read()
            for channel in channels:
                if "Intensity" in channel:
                    matrix = values[:, :, i]
                    matrix = matrix * (1000/int(exposure_time[0]))
                    std = matrix.std()
                    mean = matrix.mean()
                    data.append(std)
                    data.append(mean)
                elif "Phase Lifetime" in channel:
                    matrix = values[:, :, i]
                    std = matrix.std()
                    mean = matrix.mean()
                    data.append(std)
                    data.append(mean)
                elif "Modulation Lifetime" in channel:
                    matrix = values[:, :, i]
                    std = matrix.std()
                    mean = matrix.mean()
                    data.append(std)
                    data.append(mean)
                i = i + 1

        all_data.append(data)

    columns = ["art", "intensity_std", "intensity_mean", "phase_std", "phase_mean", "mod_std", "mod_mean"]
    all_data = pd.DataFrame(data=all_data, columns=columns)
    save_path = "csv/"
    if "488" in files[0]:
        all_data.to_csv(save_path + "dataset_488_altholz.csv")
    elif "445" in files[0]:
        all_data.to_csv(save_path + "dataset_445_altholz.csv")
    elif "405" in files[0]:
        all_data.to_csv("dataset_405_altholz_5classes.csv")

    javabridge.kill_vm()


#files = collect_plt_mlt_intensity()
#create_dataset(files)
create_dataset_from_nd2()
#create_dataset_from_nd2_altholz()