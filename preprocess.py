import numpy as np
import cv2
import pickle
import nibabel as nib
import argparse

import numpy as np

def relu(x):
    return np.maximum(0,x) # element-wise max

def preprocess_input(img, args):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    length = len(img)
    
    img = relu(img) # 검은색 픽셀 값: 음수 -> 0
    # [0,255]로 정규화 & 색 대비 최대한 유지
    for idx in range(length):
        img[idx] = img[idx] / (np.max(img[idx])/255)

    if args.process_type != "original":
        
        img = img.astype(np.uint8)

        for idx in range(length):
            # 양방향 가우시안 필터: 노이즈 제거 & edge 유지
            img[idx] = cv2.bilateralFilter(img[idx],d=-1, sigmaColor=3, sigmaSpace=5) # sigmaColor: 픽셀 간 색 차이 가우시안에 대한 표준편차, sigmaSpace: 일반 가우시안에서의 sigma와 동일

            if args.process_type == "clahe":
                # adaptive Histogram Equalization: 색 대비 강조
                img[idx] = clahe.apply(img[idx])
            else:
                # original hist equal. : 확실한 색 대비 강조
                img[idx] = cv2.equalizeHist(img[idx])


    return img


def preprocess_label(img):
    return img.astype(np.float32) # np.int64 -> np.int8 (0, 1 값만 존재하므로 용량 줄이기 // uint8로 저장하면 torchvision.transforms.ToTensor함수에서 scaling이 적용됨)


def argparser():
    parser = argparse.ArgumentParser(description='Parser for preprocessing kits21')

    parser.add_argument('--type', '-t', action='store', dest='process_type', default='clahe',
                        help="directory path to load raw image")
    args = parser.parse_args()

    return args        


def main():
    args = argparser()
    
    Logic = ["AND","MAJ","OR"]

    # nifti 개수 = 300
    for i in range(150):
        idx = str(i).zfill(3)
        proxy = nib.load(f"./kits21/kits21/data/case_00{idx}/imaging.nii.gz")
        input = proxy.get_fdata()
        input = preprocess_input(input, args)
        with open(f'./data/{args.process_type}/input/imaging_{idx}.pickle', 'wb') as f:
            pickle.dump(input, f, pickle.HIGHEST_PROTOCOL)

        for logic in Logic:
            proxy = nib.load(f"./kits21/kits21/data/case_00{idx}/aggregated_{logic}_seg.nii.gz")
            label = proxy.get_fdata()
            label = preprocess_label(label)
            with open(f'./data/{args.process_type}/label/aggregated_{idx}_{logic}.pickle', 'wb') as f:
                pickle.dump(label, f, pickle.HIGHEST_PROTOCOL)

        print(f"{i}-th process finished.")

if __name__ == '__main__':
    main()
