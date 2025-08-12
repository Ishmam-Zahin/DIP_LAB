import matplotlib.pyplot as plt
import cv2
plt.figure(figsize=(15, 7))

c = 150

def main():
    img_path = '/home/zahin/DIP/images/img.jpg'
    img_path2 = '/home/zahin/DIP/images/img2.png'
    img = cv2.imread(img_path)
    img2 = cv2.imread(img_path2)

    # img = cv2.resize(img, (100, 100))
    # img2 = cv2.resize(img2, (100, 100))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    

    # img_rgb[:100, :100] = [0, 0, 0]

    for i in range(2):
        for x in range(100):
            for y in range(100):
                img_rgb[x][y][i] = 0

    plt.imshow(img_rgb)


    plt.show()



if __name__ == '__main__':
    main()