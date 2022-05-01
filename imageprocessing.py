import PIL

print('Pillow Version:', PIL.__version__)


from PIL import Image
image = Image.open('numberone.png')
image.show()
