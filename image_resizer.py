from PIL import Image
import glob, os

#size = 128, 128

def resize_image_in_folder(folder_path, file_type, new_width):
    for infile in glob.glob(os.path.join(folder_path, '*.' + file_type)):
        try:
            file, ext = os.path.splitext(infile)
            new_name = file + "." + file_type
            print("Resizing Image : %s to %s" % (infile,new_name) )
            im = Image.open(infile)
            width, height = im.size
            if width <= new_width:
                continue
            ratio = float(width/height)
            new_height = new_width / ratio
            im.thumbnail((new_width, new_height))
            im.save(new_name, "JPEG")
        except Exception:
            try:
                os.remove(infile)
            except Exception:
                print("Cannot delete file")

def resize_image_in_folder_to_percent(folder_path, file_type, percent):
    for infile in glob.glob(os.path.join(folder_path, '*.' + file_type)):
        file, ext = os.path.splitext(infile)
        new_name = file + "." + file_type
        print("Resizing Image : %s to %s" % (infile,new_name) )
        try:
            im = Image.open(infile)
            width, height = im.size
            im.thumbnail( ( int(width*percent/100),int(height*percent/100) ) ) 
            im.save(new_name, "JPEG")
        except Exception:
            os.remove(infile)

def resize_in_all_sub_folders(path):
    folders = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    for folder in folders:
        print("Resizing for Folder : %s" % folder)
        resize_image_in_folder(folder, 'jpg', 640)

path = r'E:\Downloads\NewFolder\KNN Train'
resize_in_all_sub_folders(path)