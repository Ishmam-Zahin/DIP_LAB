import matplotlib.pyplot as plt

plt.figure(figsize=(15, 7))

def main():
    img_path = '/home/zahin/DIP/images/img2.png'
    img = plt.imread(img_path)
    img2 = img.max() - img

    plt.subplot(2, 3, 1)
    plt.imshow(img)

    plt.subplot(2, 3, 2)
    plt.imshow(img[:, :, 0], cmap='grey')

    plt.subplot(2, 3, 3)
    plt.imshow(img[:, :, 1])

    plt.subplot(2, 3, 4)
    plt.imshow(img[:, :, 2])

    plt.subplot(2, 3, 5)
    plt.imshow(img2)

    print(img.max())




    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()