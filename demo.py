import nibabel as nib
import cv2
import numpy as np

'''
https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h -> nifti file의 header에 대해 설명된 페이지
'''

def demo(proxy, idx, path):

    # Load Header
    header = proxy.header
    # print(header)

    # 원하는 헤더 불러오기
    # header_size = header['sizeof_hdr']

    # 전체 이미지 array 불러오기
    # arr = proxy.get_fdata() # fdata = full data (?) / get_fdata() == dataobj[...] 
    # print(arr.shape) # (611,512,512) -> (512,512) slice 흑백 이미지가 611개

    # 원하는 부분 이미지만 불러오기
    # for index in [200,300]:
    index = 200
    img = proxy.dataobj[index] # (512,512) ndarray 
    pixel = []
    for i in range(512):
        for j in range(512):
            pixel.append(img[i,j])
    print(set(pixel))
    cv2.imwrite(f"./check_img/{path}_{idx}_{index}.png",img*255)

def main():
    idx = str(0).zfill(3) # 000, 001, ... , 299 3자리 숫자

    # Load Proxy
    # path = f'./kits21/data/case_00{idx}/imaging.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")


    # # 1번째 콩팥
    # path = f'./kits21/kits21/data/case_00{idx}/segmentations/kidney_instance-1_annotation-1.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # path = f'./kits21/kits21/data/case_00{idx}/segmentations/kidney_instance-1_annotation-2.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # path = f'./kits21/kits21/data/case_00{idx}/segmentations/kidney_instance-1_annotation-3.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # # 2번째 콩팥
    # path = f'./kits21/kits21/data/case_00{idx}/segmentations/kidney_instance-2_annotation-1.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # path = f'./kits21/kits21/data/case_00{idx}/segmentations/kidney_instance-2_annotation-2.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # path = f'./kits21/kits21/data/case_00{idx}/segmentations/kidney_instance-2_annotation-3.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # # 종양
    # path = f'./kits21/kits21/data/case_00{idx}/segmentations/tumor_instance-1_annotation-1.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # path = f'./kits21/kits21/data/case_00{idx}/segmentations/tumor_instance-1_annotation-2.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # path
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # # 논리 연산 취한 이미지
    # path = f'./kits21/kits21/data/case_00{idx}/aggregated_AND_seg.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    # path = f'./kits21/kits21/data/case_00{idx}/aggregated_MAJ_seg.nii.gz'
    # proxy = nib.load(path)
    # demo(proxy, idx, path.split('/')[-1].split('.')[0])
    # print(f"FINISH {path}")

    path = f'./kits21/kits21/data/case_00{idx}/aggregated_OR_seg.nii.gz'
    proxy = nib.load(path)
    demo(proxy, idx, path.split('/')[-1].split('.')[0])
    print(f"FINISH {path}")




if __name__ == '__main__':
    main()


