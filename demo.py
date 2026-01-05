import os
import cv2
import sys

from src.signature_detect.loader import Loader
from src.signature_detect.judger import Judger
from src.signature_detect.cropper import Cropper
from src.signature_detect.extractor import Extractor


def save_and_optionally_show(result: dict, out_dir: str, base_name: str) -> None:
    """
    Save the cropped signature mask (and image if present) to out_dir.
    Optionally show with cv2.imshow (commented out by default).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Prefer cropped image if provided, fallback to mask
    cropped_img = result.get("cropped_image", None)
    cropped_mask = result.get("cropped_mask", None)

    saved_paths = []

    if cropped_img is not None:
        out_img_path = os.path.join(out_dir, f"{base_name}_signature.png")
        ok = cv2.imwrite(out_img_path, cropped_img)
        if ok:
            saved_paths.append(out_img_path)
        else:
            print(f"Failed to save cropped image to: {out_img_path}")

    if cropped_mask is not None:
        # Ensure single-channel mask is saved visibly; scale if needed
        mask_to_save = cropped_mask
        # If mask is 0/255 uint8, saving is fine.
        out_mask_path = os.path.join(out_dir, f"{base_name}_signature_mask.png")
        ok = cv2.imwrite(out_mask_path, mask_to_save)
        if ok:
            saved_paths.append(out_mask_path)
        else:
            print(f"Failed to save cropped mask to: {out_mask_path}")

    if saved_paths:
        print("Saved cropped signature to:")
        for p in saved_paths:
            print("  ", p)

    # Optional: show in a window (uncomment if you want GUI popup)
    if cropped_img is not None:
        cv2.imshow("Cropped Signature", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif cropped_mask is not None:
        cv2.imshow("Cropped Signature Mask", cropped_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(file_path: str) -> None:
    loader = Loader()
    extractor = Extractor(amplfier=15)
    cropper = Cropper()
    judger = Judger()

    try:
        masks = loader.get_masks(file_path)
        is_signed = False
        positive_result = None

        for mask in masks:
            labeled_mask = extractor.extract(mask)
            results = cropper.run(labeled_mask)

            for result in results.values():
                is_signed = judger.judge(result["cropped_mask"])
                if is_signed:
                    positive_result = result
                    break

            if is_signed:
                break
        print(is_signed)

        # If signed, save/show the cropped signature
        """if is_signed and positive_result is not None:
            # Build output directory next to demo.py or under a fixed folder
            out_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "outputs"
            )
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_and_optionally_show(positive_result, out_dir, base_name)"""

    except Exception as e:
        print(e)


if __name__ == "__main__":
    file_path = None
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--file":
            file_path = sys.argv[i + 1]
    if file_path is None:
        print("Need input file")
        print("python demo.py --file my-file.pdf")
    else:
        main(file_path)
