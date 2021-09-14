import glob


def collect_plt_mlt():
    my_path = r'F:\nd2_files\488 nm x10M16 100pct Altholz_Eins'
    files = glob.glob(my_path + '/**/**/*Lifetime*.tif', recursive=True)
    intensity_files = glob.glob(my_path + '/**/**/*Intensity*.tif', recursive=True)
    files += intensity_files
    files = list(dict.fromkeys(files))
    return files


def collect_files_for_each_type(files):
    fichte_files = []
    ahorn_files = []
    eiche_files = []
    laerche_files = []
    buche_files = []
    kiefer_files = []

    for file in files:
        if "Fichte" in file:
            fichte_files.append(file)
        elif "Ahorn" in file:
            ahorn_files.append(file)
        elif "Eiche" in file:
            eiche_files.append(file)
        elif "Kiefer" in file:
            kiefer_files.append(file)
        elif "Laerche" in file:
            laerche_files.append(file)
        elif "Buche" in file:
            buche_files.append(file)

    ideal_files = []
    ideal_files.append(fichte_files)
    ideal_files.append(ahorn_files)
    ideal_files.append(eiche_files)
    ideal_files.append(laerche_files)
    ideal_files.append(buche_files)
    ideal_files.append(kiefer_files)

    return ideal_files


