import numpy as np
import cv2
import matplotlib.pyplot as plt

class DifferentialDetector:
    all_diff_histo = []
    cut_threshold = 4
    fade_threshold = 1.5
    last_delta = [3, 3, 3, 3]
    avg_diffs = []

    def get_diff(self, curr_img, next_img):
        # convert RBG to YUV
        curr_YUV = cv2.split(cv2.cvtColor(curr_img, cv2.COLOR_BGR2YCrCb))
        next_YUV = cv2.split(cv2.cvtColor(next_img, cv2.COLOR_BGR2YCrCb))

        delta_yuv = [0, 0, 0, 0]         # [Y, U, V, average]

        for i in range(3):
            num_pixels = curr_YUV[i].shape[0] * curr_YUV[i].shape[1]
            curr_YUV[i] = curr_YUV[i].astype(np.int32)
            next_YUV[i] = next_YUV[i].astype(np.int32)
            delta_yuv[i] = (np.sum((curr_YUV[i] - next_YUV[i])**2) / float(num_pixels))**(1/2)
        delta_yuv[3] = sum(delta_yuv[0:3]) / 3.0
        #delta_yuv[3] = delta_yuv[0] * delta_yuv[1] * delta_yuv[2]

        relative_delta = [0,0,0,0]
        for i in range(4):
            relative_delta[i] = delta_yuv[i] / self.last_delta[i]

        self.avg_diffs.append(relative_delta[3])
        
        self.all_diff_histo.append(relative_delta)
        self.last_delta = delta_yuv
        return relative_delta

    def detect_cut_gradient(self):
        i = 0
        while i < len(self.all_diff_histo):
            if np.max(self.all_diff_histo[i]) >= self.cut_threshold:
                print("Coupure dans la trame {}".format(i))
                i += 1
            else:
                i += 1

                #val / self.last_delta * 100.0
                
    def get_metrics(self):
        return self.avg_diffs

    def display_data(self):
        plt.plot(self.all_diff_histo)
        plt.hlines(self.cut_threshold, 0, len(self.all_diff_histo))
        plt.title("Methode par difference relative")
        plt.show()

