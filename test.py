import argparse
import torch
from torchvision import transforms as T
from PIL import ImageDraw

from model import faster_rcnn_resnet_50
from dataset import cctvTestDataset

def main(img_folder, model_saved_path):
    # 연산장치 지정(GPU 또는 CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset 지정
    # batch size(integer)는 연산장치 성능에 따라 자유롭게 지정할 수 있음
    test_dataset = cctvTestDataset(img_folder)
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 모델 지정
    fasterRcnn = faster_rcnn_resnet_50()
    fasterRcnn.to(device)
    check_point = torch.load(model_saved_path)
    fasterRcnn.load_state_dict(check_point)

    # Prediction 시작
    for imgs in data_loader_test:
        with torch.no_grad():
            fasterRcnn.eval()
            imgs = list(img.to(device) for img in imgs)
            predictions = fasterRcnn(imgs) #모형에 이미지를 입력한 다음 결과 값을 predictions에 저장
            for img, prediction in zip(imgs, predictions):
                img = T.ToPILImage()(img).convert('RGB') #tensor 형태의 데이터를 PIL Image 형태로 변경

                # boxes: xmin, ymin, xmax, ymax
                # scores: box에 대한 confidence 값(box 신뢰도)
                boxes, scores = prediction['boxes'], prediction['scores']
                boxes = boxes.cpu().detach().numpy()
                scores = scores.cpu().detach().numpy()

                # 개체에 box 그리기 시작
                draw = ImageDraw.Draw(img)
                for box, score in zip(boxes, scores):
                    if score > 0.95:
                        xmin, ymin, xmax, ymax = box
                        draw.rectangle([(xmin,ymin), (xmax,ymax)], outline='red')
                    else:
                        break
                # 박스가 그려진 img 출력
                img.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default='./data/가로현수막(낮)/images', help='이미지 데이터 저장 폴더 경로')
    parser.add_argument('--model_saved_path', type=str, default='./saved_models/가로현수막(낮)_faster_rcnn.pth', help='학습 모델 저장 경로')
    opt = parser.parse_args()
    main(opt.img_folder, opt.model_saved_path)