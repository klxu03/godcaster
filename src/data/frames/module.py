import cv2
from tqdm import tqdm

class RoundSplitter:
    def __init__(self, video_path: str, template_path: str):
        self.video_path = video_path
        self.template_frame = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) 

        self.sift = cv2.SIFT.create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.template_frame, None)

        if self.des1 is None:
            raise Exception("Template image is not valid")

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) 

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def detect(self, frame) -> bool:
        kp2, des2 = self.sift.detectAndCompute(frame, None)
        if des2 is None or len(des2) < 2: # too little features detected
            return False

        matches = self.flann.knnMatch(self.des1, des2, k=2)

        threshold = 0.10
        good = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append(m)

        """
        Want to draw the KNN FLANN matches. based on https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        """

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

        return len(good) >= 1

    def split(self):
        cam = cv2.VideoCapture(self.video_path)

        frame = cv2.imread('reykjavik.png', cv2.IMREAD_GRAYSCALE)
        print(f"Detect reyjkavik {self.detect(frame)}")

        counter = 0
        next_counter = 0
        SKIP_FRAMES = 25 # frames to skip after a round is detected

        NUM_FRAMES = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(NUM_FRAMES)):
            ret, frame = cam.read()

            if not ret:
                break

            counter = counter + 1
            if counter < next_counter:
                continue

            frame = frame[0:60, int(frame.shape[1]/2) - 100:int(frame.shape[1]/2) + 100]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.detect(gray):
                next_counter = counter + SKIP_FRAMES

        cam.release()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    splitter = RoundSplitter("cut.mp4", "1:39-low.png")
    splitter.split()