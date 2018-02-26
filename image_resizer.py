from PIL import Image
import glob, os

#size = 128, 128

def resize_image_in_folder(folder_path, file_type, size):
    for infile in glob.glob(os.path.join(folder_path, '*.' + file_type)):
        file, ext = os.path.splitext(infile)
        new_name = file + "." + file_type
        print("Resizing Image : %s to %s" % (infile,new_name) )
        im = Image.open(infile)
        im.thumbnail(size)
        im.save(new_name, "JPEG")

def resize_image_in_folder_to_percent(folder_path, file_type, percent):
    for infile in glob.glob(os.path.join(folder_path, '*.' + file_type)):
        file, ext = os.path.splitext(infile)
        new_name = file + "." + file_type
        print("Resizing Image : %s to %s" % (infile,new_name) )
        im = Image.open(infile)
        width, height = im.size
        im.thumbnail( ( int(width*percent/100),int(height*percent/100) ) ) 
        im.save(new_name, "JPEG")


def resize_in_all_sub_folders(path):
    folders = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    for folder in folders:
        print("Resizing for Folder : %s" % folder)
        resize_image_in_folder_to_percent(folder, 'jpg', 30)

path = r'E:\Downloads\NewFolder\BabeSource'
resize_in_all_sub_folders(path)