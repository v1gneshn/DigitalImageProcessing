from PIL import Image, ImageDraw
from math import sin, cos, pi, radians
from numpy import uint8
import cv2 as cv


# Load image:
input_image = Image.open("pisa.jpeg")
input_pixels = input_image.load()

output_image = Image.new("RGB", input_image.size)
draw = ImageDraw.Draw(output_image)

angle = 10
degree = radians(angle)
center_x = input_image.width / 2
center_y = input_image.height / 2

for x in range(input_image.width):
    for y in range(input_image.height):
        xp = int((x - center_x) * cos(degree) - (y - center_y) * sin(degree) + center_x)
        yp = int((x - center_x) * sin(degree) + (y - center_y) * cos(degree) + center_y)
        if 0 <= xp < input_image.width and 0 <= yp < input_image.height:
            draw.point((x, y), input_pixels[xp, yp])

cv.imshow(f"Rotated Pisa{angle} degrees", uint8(output_image))
cv.waitKey(0)
