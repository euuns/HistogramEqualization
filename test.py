from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


# RGB를 HSI로 변환
def RGBtoHSI(img):
    global H, S

    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x,y))

            h = math.acos((0.5*((r-g)+(r-b)))/(((((r-g)**2)+((r-b)*(g-b)))**0.5)+0.0001))
            if b > g:
                h = 360 - h
            h = h/(2*math.pi)
            s = 1 - 3*(min(r,g,b))/(r+g+b+0.0001)
            i = (r+g+b)/3
            img.putpixel((x,y), (int(h*255), int(s*255), int(i)))
    
    H = img.split()[0]
    S = img.split()[1]

    # 히스토그램에 필요한 i만 반환
    return img.split()[2]


# 히스토그램 평활화
def HistoEqual(img):
    k = 0
    Sum = 0
    total_pixels = 0
    hist = list(0 for i in range(256))
    sum_of_hist = list(0 for i in range(256))

    # 초기화
    for z in range(0, 256):
        hist[z] = 0
        sum_of_hist[z] = 0
    
    # 히스토그램 생성
    for i in range(width): 
        for j in range(height):
            k = int(img[i][j])      # 픽셀을 정수형으로
            hist[k] = hist[k] + 1   # 각 픽셀 값의 빈도수 저장

    # 누적합 계산        
    for i in range(0,256):
        Sum = Sum + hist[i]
        sum_of_hist[i] = Sum

    # 평활화 수행
    total_pixels = width * height   # 전체 픽셀수 계산
    for i in range(0,256):
        for j in range(0,256):
            k = int(img[i][j])

            img[i][j] = sum_of_hist[k]*(255.0/total_pixels)
            #sum_of_hist 누적분포함수 * 255/total_pixels 최대 명암도

    return Image.fromarray(img)



plt.figure(figsize=(15, 3))

# 이미지 불러오기
img = Image.open("test_image.jpg").convert('RGB')
width, height = img.size
plt.subplot(1,4,1)
plt.imshow(img)
plt.title("Original Image")


array_I = np.array(RGBtoHSI(img))
new_I = HistoEqual(array_I.astype(np.uint8))


# 히스토그램 평활화 시킨 HSI를 병합하여 RGB 모드로
img_merged = Image.merge("HSV", (H, S, new_I)).convert('RGB')

plt.subplot(1,4,2)
plt.imshow(img_merged)
plt.title("HSI Image Histogram Eq (converted to RGB)")


# 이미지 분포 막대 그래프 생성
plt.subplot(1,4,4)
graph_array = np.array(new_I).flatten()
plt.hist(graph_array, bins=256, density=True, alpha=0.7, color='black')

origin_array = np.array(img).flatten()
plt.hist(origin_array, bins=256, density=True, alpha=0.5, color='purple')
plt.title('Image Distribution')

plt.show()