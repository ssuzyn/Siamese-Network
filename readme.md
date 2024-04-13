# 샴네트워크 - train 및 inference 기능

## 설정 방법

> train.py / inference.py

훈련 데이터셋, 추론 데이터셋을 로드하기 위해

dset.ImageFolder(root=)에서 root의 경로를

이미지 데이터셋이 위치한 디렉토리로 설정



## 실행 방법

> python3 app.py

1) 학습

   100에포크 학습

3) 추론

   추론할 이미지 경로 입력 → 추론 (가장 거리값이 가까운 것으로 반환)
