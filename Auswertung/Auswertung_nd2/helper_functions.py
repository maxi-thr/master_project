import scipy.interpolate
import scipy.fftpack
import scipy.signal
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle


""" Die Helper Funktions wurden nur zur Erstellung der Phasorplots verwendet"""


def cos(x):
    return math.cos(x)


def sin(x):
    return math.sin(x)


def exp(x):
    return math.exp(x)


def ceil(x):
    return math.ceil(x)


f = 30000000

xCenter_Circle = 0.5
yCenter_Circle = 0
theta_Circle = np.arange(0, math.pi, 0.01)
radius_Circle = 0.5
np_cos = np.vectorize(cos)
np_sin = np.vectorize(sin)
x_Circle = radius_Circle * np_cos(theta_Circle) + xCenter_Circle
y_Circle = radius_Circle * np_sin(theta_Circle) + yCenter_Circle


def smooth(a, wsz):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(wsz, dtype=int), 'valid') / wsz
    r = np.arange(1, wsz - 1, 2)
    start = np.cumsum(a[:wsz - 1])[::2] / r
    stop = (np.cumsum(a[:-wsz:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def matprocstruct(matrix):
    hist, bin_edges = np.histogram(matrix,
                                   bins=1500)  # hist wird als [1x1000] ausgegeben, bin_edges als [1,1001]

    # Zum zusammführen müssen die bin_edges um 1 Wert am Ende gekürzt werden:
    bin_edges_shortened = bin_edges[:-1]  # alles bis auf die letzte Spalte, weil
    smoothed_hist = smooth(hist, 5)

    for index, value in enumerate(bin_edges_shortened):
        if value < 0.3:
            bin_edges_shortened[index] = 0
    for index, value in enumerate(smoothed_hist):
        if value < 100:
            smoothed_hist[index] = 0
    mat_unproc = np.array([np.transpose(bin_edges_shortened), np.transpose(smoothed_hist)])
    try:
        f = scipy.interpolate.interp1d(mat_unproc[0, ], mat_unproc[1, ], kind='cubic', fill_value="extrapolate")
    except:
        unique_x = np.unique(mat_unproc[0, ])
        length_unique_x = len(unique_x)
        length_y = len(mat_unproc[1, ])
        unique_y = mat_unproc[1, ]
        i = 0
        for y in mat_unproc[1, ]:
            if y == 0 and length_y > length_unique_x:
                unique_y = np.delete(unique_y, i)
                length_y = length_y - 1
        f = scipy.interpolate.interp1d(unique_x, unique_y, kind='cubic', fill_value="extrapolate")

    xnew = np.arange(0, 15, 0.00001)
    ynew = f(xnew)
    mat_interp = np.array([np.transpose(xnew), np.transpose(ynew)])
    dr_mat = bin_edges_shortened

    return mat_interp, dr_mat


def model(t, coeffs):
    if len(coeffs) == 4:
        return coeffs[0] + coeffs[1] * (np.exp(- 0.5 * ((t - coeffs[2])/coeffs[3])**2))
    elif len(coeffs) == 7:
        return coeffs[0] + \
               coeffs[1] * (np.exp(- 0.5 * ((t - coeffs[2])/coeffs[3])**2)) + \
               coeffs[4] * (np.exp(- 0.5 * ((t - coeffs[5])/coeffs[6])**2))
    elif len(coeffs) == 10:
        return coeffs[0] + \
               coeffs[1] * (np.exp(- 0.5 * ((t - coeffs[2])/coeffs[3])**2)) + \
               coeffs[4] * (np.exp(- 0.5 * ((t - coeffs[5])/coeffs[6])**2)) + \
               coeffs[7] * (np.exp(- 0.5 * ((t - coeffs[8]/coeffs[9])**2)))
    elif len(coeffs) == 13:
        return coeffs[0] + \
               coeffs[1] * (np.exp(- 0.5 * ((t - coeffs[2])/coeffs[3])**2)) + \
               coeffs[4] * (np.exp(- 0.5 * ((t - coeffs[5])/coeffs[6])**2)) + \
               coeffs[7] * (np.exp(- 0.5 * ((t - coeffs[8]/coeffs[9])**2))) + \
               coeffs[10] * (np.exp(- 0.5 * ((t - coeffs[11] / coeffs[12])**2)))


def residuals(coeffs, y, t):
    return y - model(t, coeffs)


def find_peak(mat, peaks_to_save_all, gcurve_all, max_peak):
    peaks, _ = scipy.signal.find_peaks(mat[:, 1], prominence=max(mat[:, 1]) / 100, height=max(mat[:, 1]) / 10)

    np_exp = np.vectorize(exp)

    if len(peaks) < 1:
        pass

    else:
        peak_x, peak_y = mat[peaks[0]]
        if peak_y < max_peak/10:
            pass
        else:
            ordinate_left = math.ceil(peak_y * 0.6)
            index_left = (np.abs(mat[:, 1] - ordinate_left)).argmin()

            index_left = np.min(index_left)

            g1stdev = sigma_one_peak(mat, peak_x, index_left)
            g1_curve_y = peak_y * (np_exp(-0.5 * ((mat[:, 0] - peak_x) / g1stdev) ** 2))
            gcurve = np.array([np.transpose(mat), np.transpose(g1_curve_y)], dtype=object)
            peaks_to_save = (peak_x, peak_y, g1stdev)
            peaks_to_save_all.append(peaks_to_save)
            gcurve_all.append(gcurve)
            g1_ges_sub = np.vstack((mat[:, 0], mat[:, 1] - g1_curve_y)).T
            find_peak(g1_ges_sub, peaks_to_save_all, gcurve_all, max_peak)
    peaks = peaks_to_save_all
    gcurve = gcurve_all
    return peaks, gcurve


def sigma_one_peak(mat, peak_x, index_left):

    mu_sigma_left = mat[index_left]
    sigma_left = peak_x - mu_sigma_left[0]
    sigma_ges = abs(sigma_left)

    return sigma_ges


def phasor_eval(peak_plt, peak_mlt, qpos_all, ppos_all, q_stdev_circ_all, p_stdev_circ_all, std_all):

    f_sample = f * (2 * math.pi)
    theta_circle_mess = np.arange(0, 2 * math.pi, 0.01)
    if len(peak_mlt) == 1 or len(peak_plt) == 1:
        phi1_circle = math.degrees(math.atan((peak_plt[0][0]*1e-9)*f_sample))
        mod1_circle = 1/math.sqrt(((peak_mlt[0][0]*1e-9)**2*f_sample**2)+1)
        p1_pos = math.sin(math.radians(phi1_circle)) * mod1_circle
        q1_pos = math.cos(math.radians(phi1_circle)) * mod1_circle
        radius1_circle = peak_plt[0][2]/6
        std_all.append(radius1_circle)
        q1_stdev_circ = radius1_circle * np_cos(theta_circle_mess) + q1_pos
        p1_stdev_circ = radius1_circle * np_sin(theta_circle_mess) + p1_pos
        qpos_all.append(q1_pos)
        ppos_all.append(p1_pos)
        q_stdev_circ_all.append(q1_stdev_circ)
        p_stdev_circ_all.append(p1_stdev_circ)

    else:
        phi1_circle = math.degrees(math.atan((peak_plt[0][0] * 1e-9) * f_sample))
        mod1_circle = 1 / math.sqrt(((peak_mlt[0][0] * 1e-9) ** 2 * f_sample ** 2) + 1)
        p1_pos = math.sin(math.radians(phi1_circle)) * mod1_circle
        q1_pos = math.cos(math.radians(phi1_circle)) * mod1_circle
        radius1_circle = peak_plt[0][2] / 6
        std_all.append(radius1_circle)
        q1_stdev_circ = radius1_circle * np_cos(theta_circle_mess) + q1_pos
        p1_stdev_circ = radius1_circle * np_sin(theta_circle_mess) + p1_pos
        qpos_all.append(q1_pos)
        ppos_all.append(p1_pos)
        q_stdev_circ_all.append(q1_stdev_circ)
        p_stdev_circ_all.append(p1_stdev_circ)
        peak_plt.pop(0)
        peak_mlt.pop(0)
        phasor_eval(peak_plt, peak_mlt, qpos_all, ppos_all, q_stdev_circ_all, p_stdev_circ_all, std_all)

    return qpos_all, ppos_all, q_stdev_circ_all, p_stdev_circ_all, std_all


def save_dictionary(dict):
    with open('dictionary_matrix.pickle', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)


def plot_gcurve(peaks, gcurve, mat, name):

    if len(peaks) == 1:
        plt.xlim(peaks[0][0] - 6 * peaks[0][2], peaks[0][0] + 6 * peaks[0][2])
        plt.ylim(0, peaks[0][1] + 10000)
        plt.title(name)
        plt.plot(gcurve[0][0, ][0, :], gcurve[0][1, ], linewidth=2, label='gaussian curve')
        plt.plot(peaks[0][0], peaks[0][1], marker='x', markersize=10)
        plt.plot(mat[:, 0], mat[:, 1], linewidth=2, color='red', label='histogram')
        plt.legend()
        plt.xlabel('Fluoreszenzabklingzeit [ns]')
        plt.ylabel('Absolute Häufigkeit')
        plt.box(True)
        img = name + ".png"
        img_path = os.path.join("nd2_files" + '\\' + img)
        plt.savefig(img_path)
        plt.close()
        print("plot_gcurve" + name)
    else:
        plt.xlim(peaks[0][0] - 6 * peaks[0][2], peaks[0][0] + 6 * peaks[0][2])
        plt.ylim(0, peaks[0][1] + 10000)
        plt.title(name)
        for i in range(len(peaks)):
            plt.plot(gcurve[i][0, ][0, :], gcurve[i][1, ], linewidth=2, label='gaussian curve peak ' + str(i))
            plt.plot(peaks[i][0], peaks[i][1], marker='x', markersize=10)
        plt.plot(mat[:, 0], mat[:, 1], linewidth=2, color='red', label='histogram')
        plt.legend()
        plt.xlabel('Fluoreszenzabklingzeit [ns]')
        plt.ylabel('Absolute Häufigkeit')
        plt.box(True)
        img = name + ".png"
        img_path = os.path.join("nd2_files" + '\\' + img)
        plt.savefig(img_path)
        plt.close()
        print("plot_gcurve" + name)


def plot_phasor(qpos, ppos, q_stdev_circ, p_stdev_circ, name):
    if len(qpos) == 1:
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Q(W)')
        plt.ylabel('P(W)')
        plt.box(True)
        plt.plot(x_Circle, y_Circle, 'k', linewidth=2.0)
        plt.scatter(x=q_stdev_circ[0], y=p_stdev_circ[0], cmap='RdYlGn', label=name)
        plt.scatter(abs(qpos[0]), abs(ppos[0]), color="red", marker='s', s=1)
        plt.legend()
        img = "phasor_plot_" + name + ".png"
        img_path = os.path.join("nd2_files" + '\\' + img)
        plt.savefig(img_path)
        plt.close()
        print("plot_Phasor" + name)
    else:
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Q(W)')
        plt.ylabel('P(W)')
        plt.box(True)
        plt.plot(x_Circle, y_Circle, 'k', linewidth=2.0)
        for i in range(len(qpos)):
            plt.scatter(x=q_stdev_circ[i], y=p_stdev_circ[i], cmap='RdYlGn', label=name + 'peak ' + str(i))
            plt.scatter(abs(qpos[i]), abs(ppos[i]), color="red", marker='s', s=1)
        plt.legend()
        img = "phasor_plot_" + name + ".png"
        img_path = os.path.join("nd2_files" + '\\' + img)
        plt.savefig(img_path)
        plt.close()
        print("plot_Phasor" + img)
