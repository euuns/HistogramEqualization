from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


plt.figure(figsize=(15, 4))

# 이미지 불러오기
img = Image.open(r"E:\\수업\\영상처리\\히스토그램평활화\\test_image.jpg",mode='r').convert('RGB')
width, height = img.size
plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original RGB Image")

# RGB이미지를 HSI로 변환
def RGBtoHSI(img):
    global H, S

    hsi_img = img
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x,y))
            h = math.acos((0.5*((r-g)+(r-b)))/(((((r-g)**2)+((r-b)*(g-b)))**0.5)+0.0001))
            if b > g:
                h = 360 - h
            h = h/(2*math.pi)
            s = 1 - 3*(min(r,g,b))/(r+g+b+0.0001)
            i = (r+g+b)/3
            hsi_img.putpixel((x,y), (int(h*255), int(s*255), int(i)))
    
    H = hsi_img.split()[0]
    S = hsi_img.split()[1]

    # 히스토그램에 필요한 i만 반환
    return hsi_img.split()[2]


I = RGBtoHSI(img)



# 이미지를 리스트로 변환
img_array = []
def img2array():
    global img_array

    for y in range(height):
        row = []
        for x in range(width):
            row.append(I.getpixel((x,y)))
        img_array.append(row)
    
img2array()

#test
img_array = np.array(img_array)


# 해당 픽셀값의 개수
sample = {}
unique, count = np.unique(img_array, return_counts=True)
counts = np.cumsum(count)
sample = dict(zip(unique, counts))


# 히스토그램
def HistoEqual(m_image):
    k = 0
    Sum = 0
    total_pixels = 0
    hist = list(0 for i in range(256))
    sum_of_hist = list(0 for i in range(256))
    for z in range(0, 256): #초기화 단계
        hist[z] = 0
        sum_of_hist[z] = 0
    
    for i in range(width): 
        for j in range(height):
            k = int(m_image[i][j])
            hist[k] = hist[k] + 1
    for i in range(0,256):
        Sum = Sum + hist[i]
        sum_of_hist[i] = Sum

    total_pixels = 256 * 256

    for i in range(0,256):
        for j in range(0,256):
            k = int(m_image[i][j])
            m_image[i][j] = sum_of_hist[k]*(255.0/total_pixels)
            #sum_of_hist 누적분포함수 * 255/total_pixels 최대 명암도

    return m_image



new_i = HistoEqual(img_array.astype(np.uint8))
new_i = Image.fromarray(new_i)



# 이미지 병합을 위해 배열로 전환
new_I = np.array(new_i, dtype=np.uint8)
array_H = np.array(H)
array_S = np.array(S)

img_merged = Image.merge("HSV", (H, S, new_i)).convert('RGB')

plt.subplot(1,3,2)
plt.imshow(img_merged)
plt.title("HSI Image Histogram Eq (converted to RGB)")


# 이미지 분포 막대 그래프 생성
plt.subplot(1,3,3)
graph_array = new_I.flatten()
plt.hist(graph_array, bins=256, density=True, alpha=0.7, color='black')

origin_array = np.array(img).flatten()
plt.hist(origin_array, bins=256, density=True, alpha=0.5, color='purple')
plt.title('Image Distribution')

plt.show()