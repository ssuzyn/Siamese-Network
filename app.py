from src import train
from src import inference

n = int(input('1)학습 2)추론 : '))

if n == 1:
    train.run()
elif n == 2:
    img_path = input('추론 이미지 경로 입력 : ')
    inference.run(img_path)