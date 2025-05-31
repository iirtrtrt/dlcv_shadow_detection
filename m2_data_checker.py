import os


# this code count the number of datasets
def count_images(directory):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return len(
        [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
    )


shadow_train_dir = "ISTD_Dataset/train/shadow"
mask_train_dir = "ISTD_Dataset/train/mask"
test_dir = "ISTD_Dataset/test"

print("Shadow Train Images:", count_images(shadow_train_dir))
print("Mask Train Images:", count_images(mask_train_dir))
print("Test Images:", count_images(test_dir))
