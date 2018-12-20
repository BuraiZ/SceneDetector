import numpy as np
import cv2
import matplotlib.pyplot as plt

class HistogramDetector:
    all_diff_histo = []
    cut_threshold = 9000
    fade_threshold = 2000
    avg_diffs = []

    def get_histogram_diff(self, curr_img, next_img):
        # convert RBG to HSV
        curr_hsv = cv2.cvtColor(curr_img, cv2.COLOR_BGR2HSV)
        next_hsv = cv2.cvtColor(next_img, cv2.COLOR_BGR2HSV)

        #curr_hsv = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        #next_hsv = cv2.cvtColor(next_img, cv2.COLOR_BGR2RGB)

        # calculate histogram for each channel and calculate difference
        curr_histo = [0, 0, 0]      # [H, S, V]
        next_histo = [0, 0, 0]      # [H, S, V]
        diff = [0, 0, 0]         # [H, S, V, average]
        for i in range(3):
            curr_histo[i] = cv2.calcHist([curr_hsv], [i], None, [64], [0, 256])
            next_histo[i] = cv2.calcHist([next_hsv], [i], None, [64], [0, 256])
            #diff[i] = abs(cv2.compareHist(curr_histo[i],next_histo[i],0))
            diff[i] = np.sqrt(np.sum((next_histo[i][:, :] - curr_histo[i][:, :]) ** 2))

        self.avg_diffs.append(sum(diff) / 3)

        self.all_diff_histo.append(diff)
        return diff

    def detect_cut_gradient(self):
        i = 0
        while i < len(self.all_diff_histo):
            if all(value >= self.cut_threshold for value in self.all_diff_histo[i]):
                if all(value >= self.fade_threshold for value in self.all_diff_histo[i + 1]):
                    j = 2
                    while all(value >= self.fade_threshold for value in self.all_diff_histo[i + j]):
                        j += 1

                    print("Fondu entre les trames {} et {}".format(i, i + j))

                    i += j
                else:
                    print("Coupure dans la trame {}".format(i))
                    i += 1
            else:
                i += 1

    def get_metrics(self):
        #return self.all_diff_histo
        return self.avg_diffs

    def display_data(self):
        plt.plot(self.all_diff_histo)
        plt.hlines(self.cut_threshold, 0, len(self.all_diff_histo))
        plt.hlines(self.fade_threshold, 0, len(self.all_diff_histo))
        plt.title("Methode par histogramme")
        plt.show()
