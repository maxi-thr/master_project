from Klassifikation_Holzarten.Auswertung_nd2.collect_PLT_MLT import collect_plt_mlt, collect_files_for_each_type
from Klassifikation_Holzarten.Auswertung_nd2.helper_functions import *
from Klassifikation_Holzarten.Auswertung_nd2.read_nd2 import readnd2File, readnd2File_ideal
import tifffile as tiff
from PIL import Image


readnd2 = False
readnd2_ideal = True
images = []
exposure_time = []
if readnd2:
    readnd2File()

if readnd2_ideal:
    images, exposure_time = readnd2File_ideal()


# all_PLT_MLT_multi = {"PLT": {}, "MLT": {}}
# all_PLT_MLT_single = {"PLT": {}, "MLT": {}}
files = list(dict.fromkeys(collect_plt_mlt()))
files = sorted(files)

ideal_files = collect_files_for_each_type(files)


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


save_images(ideal_files, '../Neuronale_netze/images/405nm_x10_100pct_test/')


"""
Nachfolgender Code ist obsolete!
Wurde zu Beginn für die Erstellung von Phasorplots verwendet.
"""
"""
javabridge.start_vm(class_path=bioformats.JARS)


def interpolate_single_matrix():
    for file in files:
        with bioformats.ImageReader(file) as reader:
            matrix = reader.read()
            name_data = os.path.basename(file)
            name_data_struct = name_data.split('.')[0]

            "Interpolation"
            "Nur für single plots verwenden"
            mat_interp, dr_mat = matprocstruct(matrix)
            if "Phase" in name_data:
                all_PLT_MLT_single["PLT"].update({name_data_struct: {"mat_plt_x": mat_interp[0,].tolist(),
                                                                     "mat_plt_y": mat_interp[1,].tolist(),
                                                                     "dr_mat": dr_mat.tolist()}})
            else:
                all_PLT_MLT_single["MLT"].update({name_data_struct: {"mat_mlt_x": mat_interp[0,].tolist(),
                                                                     "mat_mlt_y": mat_interp[1,].tolist(),
                                                                     "dr_mat": dr_mat.tolist()}})

            reader.close()


def read_matrix_multi():
    for type in ideal_files:
        for file in type:
            with bioformats.ImageReader(file) as reader:
                matrix = reader.read()
                name_data = os.path.basename(file)
                name_data_struct = name_data.rsplit('_', 6)[0]

                "Für mehrfach plots"

                if "Phase" in name_data:
                    all_PLT_MLT_multi["PLT"].update({name_data_struct: {"matrix": matrix}})
                else:
                    all_PLT_MLT_multi["MLT"].update({name_data_struct: {"matrix": matrix}})

                reader.close()


read_matrix_multi()
save_dictionary(all_PLT_MLT_multi)


def create_list():
    collected_types = list(all_PLT_MLT_multi["PLT"].keys())
    for index, value in enumerate(collected_types):
        collected_types[index] = value.rsplit("_", 1)[0]
    types = sorted(list(set(collected_types)))
    return types


types = create_list()
all_plt = list(all_PLT_MLT_single["PLT"].keys())
all_mlt = list(all_PLT_MLT_single["MLT"].keys())

all_matrix_plt = list(all_PLT_MLT_multi["PLT"].keys())

network_data = {}


def create_mat_ges():
    for type in types:
        mat_plt_ges = np.empty(shape=(1004, 1008))
        mat_mlt_ges = np.empty(shape=(1004, 1008))
        type_list = []
        for plt in all_matrix_plt:
            if type in plt:
                type_list.append(plt)
        y = 0
        while y < len(type_list):
            mat_plt = all_PLT_MLT_multi["PLT"][type_list[y]]["matrix"]
            mat_mlt = all_PLT_MLT_multi["MLT"][type_list[y]]["matrix"]

            if y == 0:
                mat_plt_ges = mat_plt
                mat_mlt_ges = mat_mlt
            mat_plt_ges = np.concatenate((mat_plt_ges, mat_plt), 1)
            mat_mlt_ges = np.concatenate((mat_mlt_ges, mat_mlt), 1)
            y +=1
        tau_plt = mat_plt_ges.mean()
        tau_mlt = mat_mlt_ges.mean()
        gs_plt = mat_plt_ges.std()
        network_data.update({type: {"TAU_PLT": tau_plt,
                                    "TAU_MLT": tau_mlt,
                                    "GS_PLT": gs_plt}})

        mat_plt_ges_int, dr_mat_plt_ges = matprocstruct(mat_plt_ges)
        mat_mlt_ges_int, dr_mat_mlt_ges = matprocstruct(mat_mlt_ges)

        all_PLT_MLT_multi["PLT"].update({type: {"mat_plt_x": mat_plt_ges_int[0, ].tolist(),
                                                "mat_plt_y": mat_plt_ges_int[1, ].tolist(),
                                                "dr_mat": dr_mat_plt_ges.tolist()}})
        all_PLT_MLT_multi["MLT"].update({type: {"mat_mlt_x": mat_mlt_ges_int[0, ].tolist(),
                                                "mat_mlt_y": mat_mlt_ges_int[1, ].tolist(),
                                                "dr_mat": dr_mat_mlt_ges.tolist()}})


create_mat_ges()

""Plot Histogram und Phasor für Mulit-Matrix""
for i in range(len(types)):
    mat_plt_x = all_PLT_MLT_multi["PLT"][types[i]]["mat_plt_x"]
    mat_plt_y = all_PLT_MLT_multi["PLT"][types[i]]["mat_plt_y"]
    mat_plt_dr = all_PLT_MLT_multi["PLT"][types[i]]["dr_mat"]
    mat_mlt_x = all_PLT_MLT_multi["MLT"][types[i]]["mat_mlt_x"]
    mat_mlt_y = all_PLT_MLT_multi["MLT"][types[i]]["mat_mlt_y"]
    mat_mlt_dr = all_PLT_MLT_multi["MLT"][types[i]]["dr_mat"]

    mat_plt = dict(zip(mat_plt_x, mat_plt_y))
    mat_plt = np.array(list(mat_plt.items()))
    mat_mlt = dict(zip(mat_mlt_x, mat_mlt_y))
    mat_mlt = np.array(list(mat_mlt.items()))

    peaks_to_save_all = []
    gcurve_all = []
    peaks, _ = scipy.signal.find_peaks(mat_plt[:, 1], prominence=max(mat_plt[:, 1]) / 100, height=max(mat_plt[:, 1]) / 10)
    max_peak = []
    for peak in peaks:
        x, y = mat_plt[peak]
        max_peak.append(y)
    max_peak = max(max_peak)
    peaks_plt, gcurve_plt = find_peak(mat_plt, peaks_to_save_all, gcurve_all, max_peak)
    peaks_to_save_all = []
    gcurve_all = []
    peaks, _ = scipy.signal.find_peaks(mat_mlt[:, 1], prominence=max(mat_mlt[:, 1]) / 100, height=max(mat_mlt[:, 1]) / 10)
    max_peak = []
    for peak in peaks:
        x, y = mat_mlt[peak]
        max_peak.append(y)
    max_peak = max(max_peak)
    peaks_mlt, gcurve_mlt = find_peak(mat_mlt, peaks_to_save_all, gcurve_all, max_peak)

    plot_gcurve(peaks_plt, gcurve_plt, mat_plt, types[i])

    qpos_all = []
    ppos_all = []
    q_stdev_circ_all = []
    p_stdev_circ_all = []
    std_all = []
    qpos, ppos, q_stdev_circ, p_stdev_circ, std = phasor_eval(peaks_plt, peaks_mlt, qpos_all, ppos_all,
                                                         q_stdev_circ_all, p_stdev_circ_all, std_all)
    plot_phasor(qpos, ppos, q_stdev_circ, p_stdev_circ, types[i])

    network_data.update({types[i]: {"GS_PLT_TEST": std,
                                    "Qpos": qpos,
                                    "Ppos": ppos}})


""Plot Histogram und Phasor für Single-Matrix""

counts = len(all_plt)

for i in range(counts):
    mat_plt_x = all_PLT_MLT_single["PLT"][all_plt[i]]["mat_plt_x"]
    mat_plt_y = all_PLT_MLT_single["PLT"][all_plt[i]]["mat_plt_y"]
    mat_plt_dr = all_PLT_MLT_single["PLT"][all_plt[i]]["dr_mat"]
    mat_mlt_x = all_PLT_MLT_single["MLT"][all_mlt[i]]["mat_mlt_x"]
    mat_mlt_y = all_PLT_MLT_single["MLT"][all_mlt[i]]["mat_mlt_y"]
    mat_mlt_dr = all_PLT_MLT_single["MLT"][all_mlt[i]]["dr_mat"]

    mat_plt = dict(zip(mat_plt_x, mat_plt_y))
    mat_plt = np.array(list(mat_plt.items()))
    mat_mlt = dict(zip(mat_mlt_x, mat_mlt_y))
    mat_mlt = np.array(list(mat_mlt.items()))
    peaks_to_save_all = []
    gcurve_all = []
    peaks_plt, _ = scipy.signal.find_peaks(mat_plt[:, 1], prominence=max(mat_plt[:, 1]) / 100,
                                           height=max(mat_plt[:, 1]) / 10)
    peaks_plt, gcurve_plt = find_peak(mat_plt, mat_plt_dr, peaks_plt, peaks_to_save_all, gcurve_all)
    peaks_to_save_all = []
    gcurve_all = []
    peaks_mlt, _ = scipy.signal.find_peaks(mat_mlt[:, 1], prominence=max(mat_mlt[:, 1]) / 100,
                                           height=max(mat_mlt[:, 1]) / 10)
    peaks_mlt, gcurve_mlt = find_peak(mat_mlt, mat_mlt_dr, peaks_mlt, peaks_to_save_all, gcurve_all)
    peaks_to_save_all = []
    gcurve_all = []

    plot_gcurve(peaks_plt, gcurve_plt, mat_plt, all_plt[i])

    qpos_all = []
    ppos_all = []
    q_stdev_circ_all = []
    p_stdev_circ_all = []
    qpos, ppos, q_stdev_circ, p_stdev_circ = phasor_eval(peaks_plt, peaks_mlt, qpos_all, ppos_all,
                                                         q_stdev_circ_all, p_stdev_circ_all)
    plot_phasor(qpos, ppos, q_stdev_circ, p_stdev_circ, all_plt[i])


print(datetime.now() - start)
javabridge.kill_vm()

""Nach erstellen der Flowchart die Zeilen in der Console kopieren und auf https://flowchart.js.org/ einfügen ""

print("_______________________")
print("Flowchart wird erstellt")
print("_______________________")
create_flowchart()"""
