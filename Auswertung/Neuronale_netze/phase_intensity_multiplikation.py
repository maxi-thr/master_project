import glob
import tifffile as tiff
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def create_phase_intensity_image():
    my_path = r'F:\nd2_files\445 nm x10 M16 100pct Idealholz'
    files = glob.glob(my_path + '/**/**/*Phase Lifetime*.tif', recursive=True)
    intensity_files = glob.glob(my_path + '/**/**/*Intensity*.tif', recursive=True)
    files = list(dict.fromkeys(files))
    intensity_files = list(dict.fromkeys(intensity_files))

    i = 0
    for file in intensity_files:
        intensity = tiff.imread(file)
        exposure_time = Image.open(file)
        exposure_time = exposure_time.filename.split('.')[0]
        exposure_time = exposure_time.split('_')[-1]
        intensity = intensity * (1000/int(exposure_time))
        intensity = np.interp(intensity, (intensity.min(), intensity.max()), (0, 10))
        phase = tiff.imread(files[i])
        name = Image.open(file)
        name = name.filename.split('.')[0]
        name = name.split('\\')[6]
        intensity_name = name[:-14]
        phase_name = Image.open(files[i])
        phase_name = phase_name.filename.split('.')[0]
        phase_name = phase_name.split('\\')[6]
        phase_and_intensity = None
        if intensity_name in phase_name:
            phase_and_intensity = np.multiply(phase, intensity)

        intensity_name = name
        name = name + "_Phase"
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

        if "488" in name and phase_and_intensity.any():
            plt.imsave('../Neuronale_netze/images/488nm_x10_100pct/' + art + '/' + name + '.png', phase_and_intensity)
            plt.imsave('../Neuronale_netze/images/488nm_x10_100pct/' + art + '/' + intensity_name + '.png', intensity)
            print(name)
        elif "445" in name and phase_and_intensity.any():
            plt.imsave('../Neuronale_netze/images/445nm_x10_100pct/' + art + '/' + name + '.png', phase_and_intensity)
            plt.imsave('../Neuronale_netze/images/445nm_x10_100pct/' + art + '/' + intensity_name + '.png', intensity)

            print(name)
        elif "405" in name and phase_and_intensity.any():
            plt.imsave('../Neuronale_netze/images/405nm_x10_100pct/' + art + '/' + name + '.png', phase_and_intensity)
            plt.imsave('../Neuronale_netze/images/405nm_x10_100pct/' + art + '/' + intensity_name + '.png', intensity)
            print(name)

        i = i + 1


def rotate_images():
    my_path = r'images/405nm_x10_100pct/train'
    files = glob.glob(my_path + '/**/*.png', recursive=True)
    files = list(dict.fromkeys(files))
    for file in files:
        image = Image.open(file)
        name = image.filename.split('.')[0]
        name = name.split('\\')[2]
        rotated_name = name + '_rotated'
        rotated_image_90 = image.rotate(90, resample=Image.BICUBIC, expand=1, fillcolor='black')
        rotated_image_180 = rotated_image_90.rotate(90, resample=Image.BICUBIC, expand=1, fillcolor='black')
        rotated_image_270 = rotated_image_180.rotate(90, resample=Image.BICUBIC, expand=1, fillcolor='black')
        if "Ahorn" in rotated_name:
            rotated_image_90.save('../Neuronale_netze/' + my_path + '/Ahorn/' + rotated_name + '_90.png')
            rotated_image_180.save('../Neuronale_netze/' + my_path + '/Ahorn/' + rotated_name + '_180.png')
            rotated_image_270.save('../Neuronale_netze/' + my_path + '/Ahorn/' + rotated_name + '_270.png')
        elif "Buche" in rotated_name:
            rotated_image_90.save('../Neuronale_netze/' + my_path + '/Buche/' + rotated_name + '_90.png')
            rotated_image_180.save('../Neuronale_netze/' + my_path + '/Buche/' + rotated_name + '_180.png')
            rotated_image_270.save('../Neuronale_netze/' + my_path + '/Buche/' + rotated_name + '_270.png')
        elif "Eiche" in rotated_name:
            rotated_image_90.save('../Neuronale_netze/' + my_path + '/Eiche/' + rotated_name + '_90.png')
            rotated_image_180.save('../Neuronale_netze/' + my_path + '/Eiche/' + rotated_name + '_180.png')
            rotated_image_270.save('../Neuronale_netze/' + my_path + '/Eiche/' + rotated_name + '_270.png')
        elif "Fichte" in rotated_name:
            rotated_image_90.save('../Neuronale_netze/' + my_path + '/Fichte/' + rotated_name + '_90.png')
            rotated_image_180.save('../Neuronale_netze/' + my_path + '/Fichte/' + rotated_name + '_180.png')
            rotated_image_270.save('../Neuronale_netze/' + my_path + '/Fichte/' + rotated_name + '_270.png')
        elif "Kiefer" in rotated_name:
            rotated_image_90.save('../Neuronale_netze/' + my_path + '/Kiefer/' + rotated_name + '_90.png')
            rotated_image_180.save('../Neuronale_netze/' + my_path + '/Kiefer/' + rotated_name + '_180.png')
            rotated_image_270.save('../Neuronale_netze/' + my_path + '/Kiefer/' + rotated_name + '_270.png')
        elif "Laerche" in rotated_name:
            rotated_image_90.save('../Neuronale_netze/' + my_path + '/Laerche/' + rotated_name + '_90.png')
            rotated_image_180.save('../Neuronale_netze/' + my_path + '/Laerche/' + rotated_name + '_180.png')
            rotated_image_270.save('../Neuronale_netze/' + my_path + '/Laerche/' + rotated_name + '_270.png')
        print(rotated_name)

