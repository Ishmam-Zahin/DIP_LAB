import cv2
import matplotlib.pyplot as plt
import numpy as np
from zahinDIP import zahin_dip as zahin

class Node:
    def __init__(self, pdf = None, intensity = None, left = None, right = None):
        self.pdf = pdf
        self.intensity = intensity
        self.left = left
        self.right = right
        self.symbol = None
    
    def __lt__(self, other):
        return self.pdf < other.pdf
    
    def isLeaf(self):
        return (self.left == None and self.right == None)


def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/peppers.png', cv2.IMREAD_GRAYSCALE)

    counts = zahin.calc_hist(img)
    pdf = zahin._calc_pdf(counts)
    nodes = []
    for i, p in enumerate(pdf):
        nodes.append(Node(pdf = p, intensity = i))
    nodes.sort()
    
    while len(nodes) > 1:
        f = nodes[0]
        s = nodes[1]
        nodes.pop(0)
        nodes.pop(0)
        nodes.append(Node(pdf = (f.pdf + s.pdf), left = f, right = s))
        nodes.sort()
    
    huffman_nodes = []
    nodes[0].symbol = ""
    while len(nodes) > 0:
        n = nodes[0]
        nodes.pop(0)
        left = n.left
        right = n.right
        if left != None:
            left.symbol = n.symbol + '1'
            if left.isLeaf():
                huffman_nodes.append(left)
            else:
                nodes.append(left)
        if right != None:
            right.symbol = n.symbol + '0'
            if right.isLeaf():
                huffman_nodes.append(right)
            else:
                nodes.append(right)
    

    original_size = img.shape[0] * img.shape[1] * 8
    huffman_size = 0

    for node in huffman_nodes:
        index = node.intensity
        tmp = len(node.symbol)
        huffman_size += (counts[index] * tmp)


    print(f'original image size: {original_size}')
    print(f'after huffman coding applied size: {huffman_size}')
    print(f'compress ratio is: {(original_size / huffman_size):.4f}')
    print(f'space save: {((1 - (huffman_size / original_size)) * 100):.2f}%')
    
    





if __name__ == '__main__':
    main()