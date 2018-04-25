import asyncio
import cv2
import functools


def load_img(image):
    print(image)
    im = cv2.imread(image)
    cv2.imshow('Async test', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


async def test():
    tiff_img = "E:\Downloads\Training Stuff\Video Training\BangMyStepmom.16.01.03.Crystal.Nicole.mp4_frame70.jpg"
    await loop.run_in_executor(None, functools.partial(load_img, 
                                                       image=tiff_img))

    return


async def numbers():
    for number in range(20):
        await asyncio.sleep(0.5)
        print(number)
    return

if __name__ == '__main__':

    loop = asyncio.get_event_loop()
    single = asyncio.gather(test(), numbers())
    print("Start")
    try:
        loop.run_until_complete(single)
    except KeyboardInterrupt:
        pass
    print("End")
    sys.exit(1)