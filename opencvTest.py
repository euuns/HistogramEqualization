# import cv2
# # matplotlib.image 를 사용하기 위해선 matplotlib 뿐만 아니라 pillow도 깔아야 한다.
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


# # 색상 범위 설정
# lower_orange = (100, 200, 200)
# upper_orange = (140, 255, 255)

# lower_green = (30, 80, 80)
# upper_green = (70, 255, 255)

# lower_blue = (0, 180, 55)
# upper_blue = (20, 255, 200)

# # 이미지 파일을 읽어온다
# img = mpimg.imread("test_image.jpg", cv2.IMREAD_COLOR)

# # BGR to HSV 변환
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # 색상 범위를 제한하여 mask 생성
# img_mask = cv2.inRange(img_hsv, lower_orange, upper_orange)

# # 원본 이미지를 가지고 Object 추출 이미지로 생성
# img_result = cv2.bitwise_and(img, img, mask=img_mask)

# # 결과 이미지 생성
# imgplot = plt.imshow(img_result)

# plt.show()





# import cv2

# # 이미지 로드
# img = cv2.imread('test_image.jpg')

# # 이미지를 HSV 색 공간으로 변환
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # H 채널 추출
# h_channel = hsv[:,:,0]

# # 결과 이미지 출력
# cv2.imshow('H Channel', hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# import numpy as np
# import cv2

# # 이미지 로드
# img = cv2.imread('test_image.jpg')

# # 이미지를 HSV 색 공간으로 변환
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# h_channel = hsv[:,:,0]
# s_channel = hsv[:,:,1]
# v_channel = hsv[:,:,2]

# hist, bins = np.histogram(v_channel.flatten(), 256, [0, 256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()
# hist_stretch = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)


# cv2.imshow("hsv", hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# import cv2
# import numpy as np

# # 이미지 열기
# img = cv2.imread('test_image.jpg')

# # HSI로 변환
# img_hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) / np.array([179, 255, 255])

# # I 채널 추출
# i_channel = img_hsi[:, :, 2]

# # 히스토그램 계산
# hist, bins = np.histogram(i_channel.flatten(), 256, [0, 1])
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max() / cdf.max()

# cdf_m = np.ma.masked_equal(cdf_normalized, 0)
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# cdf_equalized = np.ma.filled(cdf_m,0).astype('uint8')

# img_eq = cdf_equalized[img]
# img_hsi_ = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV_FULL) / np.array([179, 255, 255])
# i_channel_ = img_hsi_[:, :, 2]

# cv2.imshow('sdf',i_channel_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






# import matplotlib.pyplot as plt
# import numpy as np

# # 데이터 준비
# x = np.random.normal(0, 1, 1000)

# # 그래프 생성
# plt.hist(x, bins=20, density=True, alpha=0.7, color='skyblue')

# # 그래프 옵션 설정
# plt.title('Normal Distribution')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# # 그래프 표시
# plt.show()




import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
data = np.random.randn(1000)

# 히스토그램 생성
n, bins, patches = plt.hist(data, bins=30)

# 각 막대에 대한 레이블 추가
for i in range(len(patches)):
    plt.text(x=bins[i]+0.05, y=n[i]+5, s=f'{int(n[i])}',
             fontsize=10, color='black', ha='left', va='bottom')

# 그래프 옵션 설정
plt.title('Histogram of Data')
plt.xlabel('Data')
plt.ylabel('Frequency')

# 그래프 표시
plt.show()