자전거로 주행하면서 촬영한 mp4 영상을 이곳에 추가합니다.
고프로 액션캠으로 테스트한 결과 코덱 문제로 opencv에서 오류가 발생합니다.

코덱 문제가 발생할 경우 음소거하면 해결됩니다:
Linux: ffmpeg -i data/videos/test.mp4 -c copy -an data/videos/test_mute.mp4
Windows: ffmpeg.exe -i data/videos/test.mp4 -c copy -an data/videos/test_mute.mp4

윈도우를 위한 ffmpeg.exe는 Walk-Assistant/ffmpeg.exe에 있습니다. cmd로 실행하세요.
