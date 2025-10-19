import cv2 as cv
import numpy as np


def draw_histogram(image,color):
  hist=cv.calcHist([image],[0],None,[256],[0,256])
  hist=cv.normalize(hist,hist).flatten()
  hist_img=np.zeros((200,256,3),dtype=np.uint8)

  for x in range(256):
    val = int(hist[x]*200)
    if color=='b':
      cv.line(hist_img,(x,200),(x,200-val),(255,0,0))
    elif color=='g':
      cv.line(hist_img,(x,200),(x,200-val),(0,255,0))
    elif color=='r':
      cv.line(hist_img,(x,200),(x,200-val),(0,0,255))
  return hist_img 


def non_linear(img):
    norm=img.astype(np.float32)/255.0
    c=1.5
    log=c*np.log2(1+norm)
  
    log_img=np.uint8(cv.normalize(log,None,0,255,cv.NORM_MINMAX))
    return log_img

def show_camera():
  cap=cv.VideoCapture(0)

  while True:
    _,frame=cap.read()
    if not _:
      break

    b,g,r=cv.split(frame)
    linear_matrix=np.ones(b.shape,dtype='uint8')*10
    linear_b=cv.add(b,linear_matrix)
    linear_g=cv.add(g,linear_matrix)
    linear_r=cv.add(r,linear_matrix)

    hist_b=draw_histogram(linear_b,'b')
    hist_g=draw_histogram(linear_g,'g')
    hist_r=draw_histogram(linear_r,'r')

    nl_b_hist=draw_histogram(non_linear(b),'b')
    nl_g_hist=draw_histogram(non_linear(g),'g')
    nl_r_hist=draw_histogram(non_linear(r),'r')

    cv.imshow("Live Camera",frame)
    cv.imshow("Blue Linear Histogram",hist_b)
    cv.imshow("Green Linear Histogram",hist_g)
    cv.imshow("Red Linear Histogram",hist_r)
    
    cv.imshow("Blue Non-Linear Histogram",nl_b_hist)
    cv.imshow("Green Non-Linear Histogram",nl_g_hist)
    cv.imshow("Red Non-Linear Histogram",nl_r_hist)

     # Wait 1 ms for key press; exit on 'q'
    if cv.waitKey(1) & 0xFF==ord('q'):
      break
  
  cap.release()
  cv.destroyAllWindows()


if __name__=='__main__':
  show_camera()