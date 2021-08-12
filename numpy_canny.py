import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import paddle
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign
import time

def smooth(x):
    sigma1 = sigma2 = 1.4
    gau_sum = 0
    gaussian = np.zeros([5, 5])
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
    print(y.shape)
    y = y.reshape(img.height-4,img.width-4)
    return y
# 计算梯度幅值
def gradients(new_gray):
    """
    :type: image which after smooth
    :rtype: 
        dx: gradient in the x direction
        dy: gradient in the y direction
        M: gradient magnitude
        theta: gradient direction
    """
    W1 = new_gray.shape[0]
    H1 = new_gray.shape[1]
    
    W_A = np.delete(new_gray,[0],axis = 0)
    W_A = np.delete(W_A,[H1-1],axis = 1)
    W_B = np.delete(new_gray,[W1-1],axis = 0)
    W_B = np.delete(W_B,[H1-1],axis = 1)

    H_A = np.delete(new_gray,[0],axis = 1)
    H_A = np.delete(H_A,[W1-1],axis = 0)
    H_B = np.delete(new_gray,[H1-1],axis = 1)
    H_B = np.delete(H_B,[W1-1],axis = 0)

    dx = np.zeros([W1-1, H1-1])
    dy = np.zeros([W1-1, H1-1])
    d = np.zeros([W1-1, H1-1])

    
    dx = np.subtract(W_A,W_B)
    dy = np.subtract(H_A,H_B)
    d = np.sqrt(np.square(dx)+np.square(dy))

    return dx, dy, d

def NMS(d, dx, dy):
    W2 = d.shape[0]
    H2 = d.shape[1]
    NMS = np.copy(d)
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
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
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
                    weight = np.abs(gradY) / np.abs(gradX)
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
            elif ((NMS[i-1, j-1:j+1] < TH).any() or (NMS[i+1, j-1:j+1]).any() 
                or (NMS[i, [j-1, j+1]] < TH).any()):
                DT[i, j] = 1

                    
        
    return DT 

img = Image.open('a.jpg')

x=img.convert('L')      #灰度化
new_gray = smooth(x)
dx,dy,d = gradients(new_gray)
NMS = NMS(d,dx,dy)
out = double_threshold(NMS)

im_out = out * 255
im_out = im_out.astype('uint8')
im_out = Image.fromarray(im_out)
im_out.save("aout.jpg")


plt.figure(figsize=(20, 10))
f = plt.subplot(121)
f.set_title('input image', fontsize=15)
plt.imshow(img)
f = plt.subplot(122)
f.set_title('output feature map', fontsize=15)
plt.imshow(out, cmap='gray')
plt.show()