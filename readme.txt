- python 환경: python 3.7 64bit

- 설치 방법
 1. pip로 설치하는 경우(global)
  - 프로젝트 폴더로 이동(ex. cd C:/python_project/faster_rcnn)
  - pip install requirements.txt
  - pip install whl/torch-1.9.1+cu102-cp37-cp37m-win_amd64.whl
  - pip install whl/torchvision-0.10.1+cu102-cp37-cp37m-win_amd64.whl

 2. pipenv로 설치하는 경우(local venv)
    - 프로젝트 폴더로 이동(ex. cd C:/python_project/faster_rcnn)
    - pipenv shell --python 3.7
    - pipenv install
    - pipenv install whl/torch-1.9.1+cu102-cp37-cp37m-win_amd64.whl
    - pipenv install whl/torchvision-0.10.1+cu102-cp37-cp37m-win_amd64.whl

- 실행 방법
 - 프로젝트 폴더로 이동(ex. cd C:/python_project/faster_rcnn)
 - python test.py --img_folder ./data/가로현수막(낮)/images --model_saved_path ./saved_models/가로현수막(낮)_faster_rcnn.pth
 - --img_folder 에 대한 default 값은 ./data/가로현수막(낮)/images
 - --model_saved_path에 대한 default 값은 ./saved_models/가로현수막(낮)_faster_rcnn.pth