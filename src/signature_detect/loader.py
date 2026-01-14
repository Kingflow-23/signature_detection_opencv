import os
import cv2
import fitz
import numpy as np

from typing import Any


class Loader:
    """
    Load an image or a pdf file.

    Attributes
    ----------
    low_threshold: tuple
        The low threshold of cv2.inRange
    high_threshold: tuple
        The high threshold of cv2.inRange

    Methods
    -------
    get_masks(path: str) -> list
        It read an image or a pdf file page by page.
        It returns the masks that the bright parts are marked as 255, the rest as 0.
    """

    def __init__(self, low_threshold=(0, 0, 250), high_threshold=(255, 255, 255)):
        if self._is_valid(low_threshold):
            self.low_threshold = low_threshold
        if self._is_valid(high_threshold):
            self.high_threshold = high_threshold

    def __str__(self) -> str:
        s = "\nLoader\n==========\n"
        s += "low_threshold = {}\n".format(self.low_threshold)
        s += "high_threshold = {}\n".format(self.high_threshold)
        return s

    def _is_valid(self, threshold: tuple) -> bool:
        if type(threshold) is not tuple:
            raise Exception("The threshold must be a tuple.")
        if len(threshold) != 3:
            raise Exception("The threshold must have 3 item (h, s, v).")
        for item in threshold:
            if item not in range(0, 256):
                raise Exception("The threshold item must be in the range [0, 255].")
        return True

    def get_masks(self, path) -> list:
        basename = os.path.basename(path)
        dn, dext = os.path.splitext(basename)
        ext = dext[1:].lower()
        if ext == "pdf":
            self.document_type = "PDF"
        elif ext == "jpg" or ext == "jpeg" or ext == "png" or ext == "tif":
            self.document_type = "IMAGE"
        else:
            raise Exception("Document should be jpg/jpeg, png, tif or pdf.")

        if self.document_type == "IMAGE":
            loader = _ImageWorker(self.low_threshold, self.high_threshold)
            return [loader.get_image_mask(path)]

        if self.document_type == "PDF":
            loader = _PdfWorker(self.low_threshold, self.high_threshold)
            return loader.get_pdf_masks(path)


class _ImageWorker:
    def __init__(self, low_threshold: tuple, high_threshold: tuple) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def make_mask(self, image) -> Any:
        """
        create a mask that the bright parts are marked as 255, the rest as 0.

        params
        ------
        image: numpy array

        return
        ------
        frame_threshold: numpy array
        """
        frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(
            frame_HSV, self.low_threshold, self.high_threshold
        )
        return frame_threshold

    def get_image_mask(self, path: str) -> Any:
        image = cv2.imread(path)
        return self.make_mask(image)


class _PdfWorker(_ImageWorker):
    def __init__(self, low_threshold, high_threshold):
        super().__init__(low_threshold, high_threshold)

    def get_pdf_images(self, path: str) -> list:
        """
        Render each page of the PDF to an image (BGR for OpenCV) at ~200 DPI.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")

        imgs = []
        # 72 points = 1 inch -> zoom = dpi / 72
        dpi = 200
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        try:
            with fitz.open(path) as doc:
                for page in doc:
                    # Render RGB pixmap; alpha disabled
                    pix = page.get_pixmap(
                        matrix=mat, colorspace=fitz.csRGB, alpha=False
                    )
                    # Convert pix.samples (bytes) -> numpy array (H, W, 3) in RGB
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, 3
                    )
                    # Convert RGB -> BGR for OpenCV
                    bgr = arr[:, :, ::-1].copy()
                    imgs.append(bgr)
        except Exception as e:
            raise RuntimeError(f"Error while rendering PDF pages with PyMuPDF: {e}")
        return imgs

    def get_pdf_masks(self, path: str) -> list:
        """
        Create masks (bright parts in 255) page by page.
        """
        images = self.get_pdf_images(path)
        masks = []
        for bgr in images:
            mask = self.make_mask(
                bgr
            )  # make_mask expects BGR and converts to HSV internally
            masks.append(mask)
        return masks
