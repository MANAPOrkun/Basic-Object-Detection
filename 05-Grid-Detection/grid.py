import cv2
import matplotlib.pyplot as plt

flat_chess = cv2.imread('02-Corner-Detection/flat_chessboard.png')

found, corners = cv2.findChessboardCorners(flat_chess,(7,7))

flat_chess_copy = flat_chess.copy()
cv2.drawChessboardCorners(flat_chess_copy, (7, 7), corners, found)

plt.imshow(flat_chess_copy)
plt.show()

