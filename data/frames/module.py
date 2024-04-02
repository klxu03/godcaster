import cv2
import numpy as np

"""
TODO
To improve speed/performance of the round detection, we have two options:

- Crop out the video so it's just the 350px x 100px region around the timer so it's faster to process
- OCR on the image to detect the time length and see if it matches 1:39, if this is faster 
"""

class RoundSplitter:
    def __init__(self, video_path: str, template_path: str):
        self.video_path = video_path
        self.template_frame = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) 

        self.sift = cv2.SIFT.create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.template_frame, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

    def detect(self, frame, counter):
        kp2, des2 = self.sift.detectAndCompute(frame, None)
        matches = self.flann.knnMatch(self.des1, des2, k=2)

        threshold = 0.15
        good = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append(m)

        print(f"Len good {len(good)}")

        """
        Want to draw the KNN FLANN matches. based on https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

        if len(good):
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]

            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < threshold*n.distance:
                    matchesMask[i]=[1,0]
            
            draw_params = dict(matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = cv2.DrawMatchesFlags_DEFAULT)
            
            img3 = cv2.drawMatchesKnn(self.template_frame,self.kp1,frame,kp2,matches,None,**draw_params)
            import matplotlib.pyplot as plt
            plt.imshow(img3,),plt.show()
            plt.imshow(frame),plt.show()
        """

        return len(good)

    def split(self):
        cam = cv2.VideoCapture(self.video_path)

        counter = 0
        found = 0

        while True:
            ret, frame = cam.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = found + (self.detect(gray, counter) >= 1)
            counter = counter + 1
            print(f"Frame {counter}, found {found} rounds")
            # break

        print(f"Found {counter} rounds")
        cam.release()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    splitter = RoundSplitter("cut.mp4", "round_start.png")
    splitter.split()