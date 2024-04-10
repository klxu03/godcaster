from captions import WhisperXRunner

def main():
    runner = WhisperXRunner(["data/frames/cut.mp4"], "data/output", compute_type='int8')
    runner.run()

if __name__ == "__main__":
    main()