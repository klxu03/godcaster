import cv2
from tqdm import tqdm
import easyocr
import math
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

class RoundSplitter:
    def __init__(self, template_path: str, output_dir: str):
        self.template_frame = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) 
        self.output_dir = output_dir

        self.sift = cv2.SIFT.create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.template_frame, None)

        if self.des1 is None:
            raise Exception("Template image is not valid")

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) 

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.reader = easyocr.Reader(['en'])

    """
    Returns 0 if round not detected, 1 if 1:39 detected, 2 if 1:38 detected

    Does an initial KNN FLANN match pass, which has about 20 fps throughput and then if it passes the threshold of existing 1:39 template matching
    Go ahead and do OCR on the frame to check if it is 1:39 or 1:38 (sometimes might just skip 1:39, but will catch 1:38, chance of failing twice is lesser)
    """
    def detect(self, frame) -> int:
        kp2, des2 = self.sift.detectAndCompute(frame, None)
        if des2 is None or len(des2) < 2: # too little features detected
            return False

        matches = self.flann.knnMatch(self.des1, des2, k=2)

        threshold = 0.175
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
            cv2.imshow(img3)
            import matplotlib.pyplot as plt
            plt.imshow(img3,),plt.show()
            plt.imshow(frame),plt.show()
        """

        if len(good) == 0:
            return 0
        
        # cv2.imwrite("frame.jpg", frame)

        res = self.reader.readtext(frame)[-1][1]
        # print("EasyOCR", res)

        if set("138") <= set(res):
            return 2

        return set("139") <= set(res)


    def split(self, video_path: str):
        vid = cv2.VideoCapture(video_path)

        counter = 0
        next_counter = 0
        FPS = vid.get(cv2.CAP_PROP_FPS) # frames to skip after a round is detected
        # print("FPS", FPS)

        round_starts = []

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

            try:
                if self.detect(gray):
                    num_rounds += 1
                    # print(f"Round detected {num_rounds}")
                    next_counter = counter + 15 # skip at least 15 seconds
                    round_starts.append(counter)
            except Exception as e:
                print(f"Error in detecting round {e}")

        vid.release()
        cv2.destroyAllWindows()
        round_starts.append(NUM_FRAMES // FPS)

        dir = video_path.split(".")[0]
        os.makedirs(dir, exist_ok=True)
        file_format = video_path.split(".")[-1]
        CLIP_MAX_LENGTH = 5 * 60 # 5 mins

        # print("Round starts", round_starts)
        print(f"For {video_path}, detected {num_rounds} rounds")
        for i in range(num_rounds - 1):
            start = round_starts[i] 
            end = round_starts[i + 1]

            if end - start > CLIP_MAX_LENGTH:
                continue

            # don't prepend output_dir on slurm since video_path includes absolute /scratch/kxu39/ already
            ffmpeg_extract_subclip(video_path, start, end, targetname=f"{dir}/round_{i}.{file_format}")

if __name__ == "__main__":
    splitter = RoundSplitter("1_39.jpg", ".")
    splitter.split("sample2.webm")