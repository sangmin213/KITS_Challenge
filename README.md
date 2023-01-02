# Kits21 Challenge

> 데이터 설명
 - imaging : nifti 이미지. (512,512) 크기의 총 611장의 슬라이스가 존재. 흑백 이미지<br/>
**Segmentation (각 pixel(voxel) 값이 0 or 1)**
 - kidney_instance-#_annotation-## : 2개의 신장 중 #번째 신장에 대한 ##번째 전문가의 segmentation (label)
 - tumor : 종양에 대한 segmenation
 - aggregated_AND/MAJ/OR : 각 논리적 연산으로 2개의 신장 segmentation을 더한 결과

> Experiment & Result
 - Kidney Segmentation
 - Train set: case000 ~ case199
 - Test set: case230 ~ case270
 - Epoch: 15
 - Training set DICE score : 0.9623
 - Test set DICE score : 0.9245