import cv2
import pydicom
def read_dicom_image(dicom_file):
  try:
    ds = pydicom.dcmread(dicom_file)
    img = ds.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype('uint8')
  except Exception as e:
    print(f"Neuspesno ucitavanje DICOM: {dicom_file}: {e}")
    return None