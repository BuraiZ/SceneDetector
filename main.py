import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from histogram import HistogramDetector
from outline import OutlineDetector
from differential import DifferentialDetector


parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, nargs='?', default="outline")
args = parser.parse_args()

frames = []
histogram_detector = HistogramDetector()
outline_detector = OutlineDetector()
differential_detector = DifferentialDetector()

def detect(file, detector):
    all_detector = True
    
    if not read_video(file):
        return
    
    for i in range(1, len(frames)):
        if (detector == "histogram" or all_detector):
            diff_histo = histogram_detector.get_histogram_diff(frames[i-1], frames[i])

        if (detector == "outline" or all_detector):
            last_gray = cv2.cvtColor( frames[i-1], cv2.COLOR_BGR2GRAY )
            curr_gray = cv2.cvtColor( frames[i], cv2.COLOR_BGR2GRAY )
            diff_outline = outline_detector.detect_outline_cut(last_gray, curr_gray)
			
        if (detector == "differential" or all_detector):
            diff_diff = differential_detector.get_diff(frames[i-1], frames[i])
    
    
    
    if (detector == "histogram" or all_detector):
        print("============================ HISTOGRAM ================================")
        histogram_detector.detect_cut_gradient()
        histogram_detector.display_data()
        histo_data = histogram_detector.get_metrics()
    

    if (detector == "outline" or all_detector):
        print("============================ OUTLINE ================================")
        outline_detector.detect_cut_gradient()
        outline_detector.display_data()
        outline_data = outline_detector.get_metrics()
	
    if (detector == "differential" or all_detector):
        print("============================ DIFFERENTIAL ================================")
        differential_detector.detect_cut_gradient()
        differential_detector.display_data()
        differential_data = differential_detector.get_metrics()

    histo_outline = np.multiply(histo_data, outline_data)
    plt.plot(histo_outline)
    plt.title("Combinaison entre histogramme et contour")
    plt.show()

    histo_differential = np.multiply(histo_data, differential_data)
    plt.plot(histo_differential)
    plt.title("Combinaison entre histogramme et difference relative")
    plt.show()

    differential_outline = np.multiply(differential_data, outline_data)
    plt.plot(differential_outline)
    plt.title("Combinaison entre difference relative et contour")
    plt.show()

    
    total_data = np.multiply(differential_outline, histo_data)
    plt.plot(total_data)
    plt.title("Combinaison entre histogramme, difference relative et contour")
    plt.show()

def read_video(file):
    capture = cv2.VideoCapture()
    capture.open(file)

    # open video
    if not capture.isOpened():
        print("Could not open the video")
        return False

    # read all frames from video
    while True:
        return_value, image = capture.read()
        if not return_value:
            break
        else:
            frames.append(image)

    capture.release()

    return True


if __name__ == "__main__":
    detect("pole-vaulter.avi", args.d)
