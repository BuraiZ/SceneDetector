import numpy as np
import cv2
import matplotlib.pyplot as plt

class OutlineDetector:
    all_pin = []
    all_pout = []
    all_pmax = []
    tc = 0.8
    tg = 0.5

    def detect_outline_cut(self, curr_img, next_img, print_info=False):
        dilatation_window_size = 6
        bin_threshold = 30      # black if higher, white if lower

        # Add 2 extra row and column to convolute on edge
        col=curr_img[:,0]
        curr_img = np.column_stack((col,curr_img))
        col=curr_img[:,len(curr_img[0])-1]
        curr_img = np.column_stack((curr_img,col))
        row=curr_img[0,:]
        curr_img = np.row_stack((row,curr_img))
        row=curr_img[len(curr_img)-1,:]
        curr_img = np.row_stack((curr_img,row))

        col=next_img[:,0]
        next_img = np.column_stack((col,next_img))
        col=next_img[:,len(next_img[0])-1]
        next_img = np.column_stack((next_img,col))
        row=next_img[0,:]
        next_img = np.row_stack((row,next_img))
        row=next_img[len(next_img)-1,:]
        next_img = np.row_stack((next_img,row))


        # sobel
        sobelx = cv2.Sobel(curr_img,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(curr_img,cv2.CV_64F,0,1,ksize=3)
        mag = np.hypot(sobelx,sobely)
        mag *= 255.0/np.max(mag)
        E_curr = mag

        sobelx = cv2.Sobel(next_img,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(next_img,cv2.CV_64F,0,1,ksize=3)
        mag = np.hypot(sobelx,sobely)
        mag *= 255.0/np.max(mag)
        E_next = mag


        # threshold outline intensity
        # ForceGradient = np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
        # aretes = ForceGradient>80
        # plt.figure()
        # plt.imshow(aretes, plt.get_cmap('binary'))
        # plt.show()


        # get binary outline
        ret, bin_img_curr = cv2.threshold(E_curr,bin_threshold,255,cv2.THRESH_BINARY)
        ret, bin_img_next = cv2.threshold(E_next,bin_threshold,255,cv2.THRESH_BINARY)
        E_curr = bin_img_curr
        E_next = bin_img_next


        # dilate the outline
        kernel = np.ones((dilatation_window_size,dilatation_window_size), np.uint8)

        img_dilation = cv2.dilate(bin_img_curr, kernel, iterations=1)
        img_dilation /= 255
        D_curr = img_dilation

        img_dilation = cv2.dilate(bin_img_next, kernel, iterations=1)
        img_dilation /= 255
        D_next = img_dilation


        # print info
        if print_info:
            fig = plt.figure()

            ax1 = fig.add_subplot(221)
            ax1.imshow(E_curr, plt.get_cmap('gray'))
            ax1.title.set_text('E_curr')

            ax2 = fig.add_subplot(222)
            ax2.imshow(E_next, plt.get_cmap('gray'))
            ax2.title.set_text('E_next')

            ax3 = fig.add_subplot(223)
            ax3.imshow(D_curr, plt.get_cmap('gray'))
            ax3.title.set_text('D_curr')

            ax4 = fig.add_subplot(224)
            ax4.imshow(D_next, plt.get_cmap('gray'))    
            ax4.title.set_text('D_next')

            plt.show()
        

        # find incoming outline (Pin)
        Pin = 1 - ((np.matrix(D_curr*E_next).sum()) / np.matrix(E_next).sum())


        # find outgoing outline (Pout)
        Pout = 1 - ((np.matrix(E_curr*D_next).sum()) / np.matrix(E_curr).sum())
        
        self.all_pin.append(Pin)
        self.all_pout.append(Pout)
        return Pin, Pout

    def display_data(self):
        plt.plot(self.all_pmax)
        plt.hlines(self.tc, 0, len(self.all_pmax))
        plt.title("Methode par contour")
        plt.show()

    def get_metrics(self):
        return self.all_pmax
    
    def detect_cut_gradient(self):
        for i in range(0, len(self.all_pin)):
            p_max = max(self.all_pin[i], self.all_pout[i])
            self.all_pmax.append(p_max)
            if p_max >= self.tc:
                print("Coupure dans la trame {}".format(i))
