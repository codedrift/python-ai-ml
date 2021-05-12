try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


print(pytesseract.image_to_string(Image.open('sipgate.webp')))

print(pytesseract.get_languages(config=''))