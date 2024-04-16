import cv2
from tqdm import tqdm
import easyocr
import math

class RoundSplitter:
    def __init__(self, template_path: str):
        self.template_frame = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) 

        self.sift = cv2.SIFT.create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.template_frame, None)

        if self.des1 is None:
            raise Exception("Template image is not valid")

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) 

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.reader = easyocr.Reader(['en'])

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
        """

        if len(good) == 0:
            return False

        res = self.reader.readtext(frame)[-1][1]
        print("EasyOCR", res)

        return set("139") <= set(res)


    def split(self, video_path: str):
        vid = cv2.VideoCapture(video_path)

        counter = 0
        next_counter = 0
        FPS = vid.get(cv2.CAP_PROP_FPS) # frames to skip after a round is detected
        print("FPS", FPS)

        NUM_FRAMES = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        num_rounds = 0
        for i in tqdm(range(math.ceil(NUM_FRAMES/FPS) + 120)):
            counter = counter + 1
            if counter < next_counter:
                continue
            vid.set(cv2.CAP_PROP_POS_FRAMES, int(counter * FPS))

            ret, frame = vid.read()
            if not ret:
                break

            frame = frame[0:80, int(frame.shape[1]/2) - 100:int(frame.shape[1]/2) + 100]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

            if self.detect(gray):
                num_rounds += 1
                print(f"Round detected {num_rounds}")
                next_counter = counter + 5 # skip at least 15 seconds

        vid.release()
        cv2.destroyAllWindows()

    def test_reykjavik(self):
        frame = cv2.imread('reykjavik.png', cv2.IMREAD_GRAYSCALE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[0:80, int(frame.shape[1]/2) - 100:int(frame.shape[1]/2) + 100]
        print(f"Detect reyjkavik {self.detect(frame)}")
        
            
if __name__ == "__main__":
    splitter = RoundSplitter("1_39.jpg")
    # splitter.test_reykjavik()
    splitter.split("sample.mkv")