# Walk-Assistant
시각 장애인을 위한 보행로 인식 (Recognizing sidewalk for the visually impaired)

![Alt Text](img/result.gif)

뉴스 참고: https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=102&oid=025&aid=0002848946

TensorFlow Korea 참고: https://www.facebook.com/groups/TensorFlowKR/permalink/748798628794531/?__xts__[0]=68.ARA6-n9squLyI-_Kj09tjlLi3v7nQOcgWBYdqUGUTIlLm38kzfLx1GGUdcfjfKrC9aX74RGXWMRYUo-yVboKFj4_zZSqjihI5YR9v5K-LPn5hss2K50S5W5DKAqhS3vDoyGzNtUdDpY3ltq2T92sLsOa4LDF2hbogJkq3lE2TKDNMBwG06OA&__tn__=CH-R

업그레이드 버전 1: https://www.facebook.com/groups/TensorFlowKR/permalink/751923961815331/

업그레이드 버전 2: https://www.facebook.com/groups/TensorFlowKR/permalink/767015970306130/

결과: https://www.facebook.com/groups/TensorFlowKR/permalink/767109076963486/


오픈소스로 공개함으로써 이 프로젝트가 하루빨리 완성될 수 있으면 좋겠습니다.

학습 데이터가 절대적으로 부족합니다! 대신 학습 데이터의 품질이 떨어지면 정확도는 기하급수적으로 감소합니다.

코드에 기여하지 않더라도 단순히 걸어다니면서 찍은 동영상을 공유하는것 만으로도 큰 도움이 됩니다!

클라우드를 지원받게 되면 공개 FTP 클라우드 서버를 만들어서 빅데이터를 모두가 같이 모을 수 있도록 하고자 합니다.


# 사용법

1. data/videos에 학습할 mp4 파일을 넣습니다. (코덱 문제가 발생할 수 있습니다. 영상이 중지될 경우 data/videos/readme.txt 를 참고하세요.)
2. python3 make_data.py 를 실행하세요. data/videos/*.mp4 파일들을 data/frames 에 분리되어서 저장됩니다. (label.txt 는 더 이상 사용되지 않습니다.)
3. python3 annotation.py 를 실행하세요. data/frames/*.jpg 파일들에 대해서 라벨링을 합니다. 결과는 data/frames/annotation.txt 에 저장됩니다.
4. python3 train.py 를 실행하세요. data/frames/*.jpg 와 data/frames/annotation.txt 가 필요합니다. model.py에서 CuDNNLSTM 를 사용합니다.
GPU가 아닌경우 build_simple_model 함수의 "x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)" 을 "x = Bidirectional(LSTM(32, return_sequences=True))(x)" 로 변경하세요.
5. python3 predict.py data/videos/test.mp4 를 실행하세요. 결과는 output/output.avi 에 저장됩니다. 추가적인 사용법은 python3 predict.py -h 를 입력하세요.

How to Annotation:

기존에 있는 초록색 박스는 인공지능이 예측한 결과입니다.

왼쪽 클릭으로 보행로 영역을 추가하세요.

오른쪽 클릭으로 보행로가 아닌 영역을 제거하세요.
