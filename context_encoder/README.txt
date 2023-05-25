1. ./data/img_align_celeba 폴더에 이미지 데이터를 넣어주세요. 
그리고 터미널 경로를 해당 폴더로 이동해주세요.
2. 다음 명령어로 패키지를 설치해 주세요.
pip install -r requirements.txt
3. 다음 명령어로 코드를 실행해 주세요.
python3 context_encoder.py
4. 폴더에 생성된 checkpoint.pt 가 학습된 모델입니다.
모델 출처 : https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/context_encoder