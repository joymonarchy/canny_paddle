import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import paddle
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign
import time

#进行高斯模糊处理
def smooth(x):
    sigma1 = sigma2 = 1.4
    gau_sum = 0
    gaussian = paddle.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp((-1/(2*sigma1*sigma2))*(np.square(i-3) 
                                    + np.square(j-3)))/(2*math.pi*sigma1*sigma2)
            gau_sum =  gau_sum + gaussian[i, j]
                    
        # 归一化处理
    gaussian = gaussian / gau_sum
    w = gaussian.reshape([1, 1, 5, 5])

    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=[5, 5], 
                weight_attr=paddle.ParamAttr(
                initializer=Assign(value=w)))
    
    x = np.array(x).astype('float32')
    x = x.reshape(1, 1, img.height, img.width)
    x = paddle.to_tensor(x)
    y = conv(x)

    y = y.numpy()
    y = y.reshape(img.height-4,img.width-4)
    y = paddle.to_tensor(y)
    return y
# 计算梯度幅值
def gradients(new_gray):

    W1 = new_gray.shape[0]
    H1 = new_gray.shape[1]
    
    W_A = new_gray[1:W1, 0:H1-1]
    W_B = new_gray[0:W1-1, 0:H1-1]

    H_A = new_gray[0:W1-1, 1:H1]
    H_B = new_gray[0:W1-1, 0:H1-1]

    dx = paddle.zeros([W1-1, H1-1])
    dy = paddle.zeros([W1-1, H1-1])
    d = paddle.zeros([W1-1, H1-1])

    dx = W_A - W_B
    dy = H_A - H_B
    d = paddle.sqrt(paddle.square(dx)+paddle.square(dy)) 

    return dx, dy, d
#非极大值抑制
def NMS(d, dx, dy):
    W2 = d.shape[0]
    H2 = d.shape[1]
    print(d.shape)
    NMS = paddle.assign(d)
    NMS[0,:] = NMS[W2-1,:] = NMS[:,0] = NMS[:, H2-1] = 0


    tic = time.time()
    for i in range(1, W2-1):
        for j in range(1, H2-1):
            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]

                # 如果Y方向幅度值较大
                if paddle.abs(gradY) > paddle.abs(gradX):
                    weight = paddle.abs(gradX) / paddle.abs(gradY)
                    grad2 = d[i-1, j]
                    grad4 = d[i+1, j]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i-1, j+1]
                        grad3 = d[i+1, j-1]
                        
                # 如果X方向幅度值较大
                else:
                    weight = paddle.abs(gradY) / paddle.abs(gradX)
                    grad2 = d[i, j-1]
                    grad4 = d[i, j+1]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i+1, j-1]
                        grad3 = d[i-1, j+1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]



                gradTemp1 = weight * grad1 + (1-weight) * grad2
                gradTemp2 = weight * grad3 + (1-weight) * grad4

                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0

    toc = time.time()

    print(toc-tic)

    return NMS
#设置高低阈值
def double_threshold( NMS):
        
    W3 = NMS.shape[0]
    H3 = NMS.shape[1]
    DT = np.zeros([W3, H3])               
    # 定义高低阈值
    TL = 0.2 * np.max(NMS)
    TH = 0.3 * np.max(NMS)
    for i in range(1, W3-1):
        for j in range(1, H3-1):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            # elif ((NMS[i-1, j-1:j+1] < TH).any() or (NMS[i+1, j-1:j+1]).any() 
            #     or (NMS[i, [j-1, j+1]] < TH).any()):
            elif (NMS[i-1, j-1:j+1].max() < TH or (NMS[i+1, j-1:j+1]).any() 
                or (NMS[i, [j-1, j+1]] < TH).any()):
                DT[i, j] = 1

                    
        
    return DT 



#打开图片
img = Image.open('a.jpg')
#对图片进行灰度化处理，降低计算开销
gray_img=img.convert('L')  
#高斯滤波处理灰度化图片，去除图像噪声，这里取常用的5x5，σ=1.4的高斯滤波器
new_gray = smooth(gray_img)
#计算梯度幅值，得到dx,dy,d
dx,dy,d = gradients(new_gray)
#进行非极大值抑制，保留局部最大梯度，增强边界
NMS = NMS(d,dx,dy)
#阈值过滤噪声需要转换成narray
NMS = NMS.numpy()
#高低阈值的作用是滤除噪声等其他因素引起的小的梯度值保留大的梯度值，这里参数可以调节
out = double_threshold(NMS)

#存储处理后的图像,将浮点数转换为uint8存储成jpg格式
im_out = out * 255
im_out = im_out.astype('uint8')
im_out = Image.fromarray(im_out)
im_out.save("aout.jpg")


#下面用plt来展示图片
plt.figure(figsize=(20, 10))
f = plt.subplot(121)
f.set_title('input image', fontsize=15)
plt.imshow(img)
f = plt.subplot(122)
f.set_title('output feature map', fontsize=15)
plt.imshow(out, cmap='gray')
plt.show()